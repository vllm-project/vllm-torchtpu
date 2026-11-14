"""GLM-5.2 decode megakernel for TP x DCP (x EP): the whole 78-layer target forward of one decode /
MTP-verify step as ONE pallas_call per rank, R ranks on one mesh axis (TP = DCP = EP = R).

Design: kernel_work/glm52-mk-tp16dcp16/WHITEBOARD.v1.md (§5 dataflow, §10 numerics). Ported from the
DP16 x EP16 kernel (kernels/glm5_megakernel/kernel.py on annyan/glm52-c16-stack7-topkroll2): the weight
ring (`pump/take/give`), stacked operands, in-kernel exchanges and the TC top-k carry over; attention
becomes split-KV over the rank's DCP shard with an LSE merge, the top-k two-level, and every rank works
on every row of the step.

Rows: TMAX = R * RPR rows per step (bucket), row t's residual owned by rank t % R. Per layer, rank `me`:
   owner: residual + input RMSNorm -> AG normed rows (all ranks hold all rows) -> replicated fused_qkv_a,
   q_a / kv_a norms, k_pe rope -> owned new KV rows written into the (aliased) caches -> q_b on this
   rank's HR heads + rope + W_UK absorb -> AG absorbed q (all 64 heads everywhere) -> [indexer layers:
   replicated indexer q / k, owned new idx records written, local scan of the rank's KV shard per request,
   local top-KT per row, AG candidate keys, global top-KT, owned winners] -> per row: gather the owned
   winners' KV rows, attention over all heads -> bf16 partial + lse -> A2A to head owners, LSE merge ->
   W_UV + o_proj on HR heads (local K, per-rank scale) -> RS bf16 partial rows to owners -> owner:
   residual + post RMSNorm -> AG -> replicated router, local experts (hits) + shared TP slice
   (dense layers: dense TP slice) -> RS -> owner.
   final RMSNorm on owners -> AG -> out [M, H] (every rank: drafter and lm_head read all rows).

Activations are [M, X] row blocks (M >= 32: an fp8 matmul operand block needs >= 32 sublane rows).
"""

import dataclasses
import functools
import math
import os

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from . import pool_alias, topk2
from .layout import rup, smalls_layout

FP8 = jnp.float8_e4m3fn
BF16 = jnp.bfloat16
F32 = jnp.float32
I32 = jnp.int32
U32 = jnp.uint32
U8 = jnp.uint8
FP8_MAX = 448.0
INT_MIN = np.int32(-(2 ** 31))
NT = (((1,), (1,)), ((), ()))
NN = (((1,), (0,)), ((), ()))
COLLECTIVE_ID = int(os.environ.get("MKD_COLLECTIVE_ID", "39"))
GATHER_UNROLL = int(os.environ.get("MKD_GUNROLL", "8"))
ABLATIONS = ("weights", "gather", "index", "attn", "experts", "coll",
             "d_lm", "d_eh", "d_emb", "d_ishare", "d_glue")   # d_*: drafter glue (timing only)


# ------------------------------------------------------------------------------------------
# Operand plan
# ------------------------------------------------------------------------------------------

def weight_kinds(cfg):
    """kind -> (layers, per-rank shape, dtype)."""
    all_l = list(range(cfg.L))
    full = [l for l in all_l if cfg.full_indexer(l)]
    moe = [l for l in all_l if cfg.is_moe(l)]
    dense = [l for l in all_l if not cfg.is_moe(l)]
    H = cfg.H
    return {
        "qkv_a": (all_l, (H, rup(cfg.QKVA)), FP8),          # replicated, N padded
        "q_b": (all_l, (cfg.QL, cfg.HR * cfg.DQ), FP8),     # this rank's HR heads' columns
        "w_uk": (all_l, (cfg.HR, cfg.DN, cfg.KL), BF16),    # this rank's heads
        "w_uv": (all_l, (cfg.HR, cfg.KL, cfg.DV), BF16),    # this rank's heads (TP)
        "o_proj": (all_l, (cfg.KR, H), FP8),                # local K rows (TP)
        "idx_wq": (full, (cfg.QL, cfg.IH * cfg.IDIM), FP8),  # replicated
        "idx_wk": (full, (H, rup(cfg.IDIM + cfg.IH)), BF16),  # replicated, (K, N) copy
        "gate": (moe, (H, cfg.E), BF16),                    # replicated, (K, N) copy
        "ew13": (moe, (cfg.ER, H, 2 * cfg.I), FP8),
        "ew2": (moe, (cfg.ER, cfg.I, H), FP8),
        "es13": (moe, (cfg.ER, 2 * cfg.I), F32),
        "es2": (moe, (cfg.ER, H), F32),
        "sw13": (moe, (H, 2 * cfg.ISR), FP8),               # TP slice
        "sw2": (moe, (cfg.ISR, H), FP8),                    # TP slice
        "w13": (dense, (H, 2 * cfg.IDR), FP8),              # TP slice
        "w2": (dense, (cfg.IDR, H), FP8),                   # TP slice
        # decode replicas (KREP > 0 only): slot q of this rank holds a copy of the kernel expert
        # its smalls table names (rep_e / rep_s), see replicas.py
        **({} if not cfg.KREP else {
            "rw13": (moe, (cfg.KREP, H, 2 * cfg.I), FP8),
            "rw2": (moe, (cfg.KREP, cfg.I, H), FP8),
            "rs13": (moe, (cfg.KREP, 2 * cfg.I), F32),
            "rs2": (moe, (cfg.KREP, H), F32),
        }),
    }


def cache_kinds(cfg, nb):
    all_l = list(range(cfg.L))
    full = [l for l in all_l if cfg.full_indexer(l)]
    pg = cfg.page
    return {
        "nope": (all_l, (nb, pg, 128), U32),
        "rope": (all_l, (nb, pg // 4, 128), U32),
        "idx": (full, (nb, cfg.IREC // 4, 4, pg), U8),
    }


@dataclasses.dataclass(frozen=True)
class Tiles:
    # "mixed": bf16 activation x fp8 weight (exact products; Mosaic's fp8 x fp8 is off ~0.4% on
    # TPU7x, j-da524a3e); "bf16": upcast both; "fp8": native.
    dot: str = "mixed"
    bank: int = 3 << 20
    nbank: int = 11          # 12 x 3 MiB overflowed VMEM by 291 KB with the double-buffered gather (AOT)
    trace: bool = False
    scan_depth: int = 8
    topk_kc: int = 512          # local extraction chunk (1-chip sweep j-ffda0844: 256 -> 512 = -3 us/row)
    topk_kc_global: int = 256   # global compaction chunk (owns ~K/R slots: one chunk with the dyn count)

    def tk(self, K, N, dtype):
        """Largest 128-multiple divisor of K whose (tk, N) tile fits one bank."""
        best = None
        for tk in range(128, K + 1, 128):
            if K % tk == 0 and pool_alias.storage_size((tk, N), jnp.dtype(dtype)) <= self.bank:
                best = tk
        assert best is not None, ("no 128-row tile fits a bank", K, N, dtype, self.bank)
        return best


# ------------------------------------------------------------------------------------------
# In-kernel helpers
# ------------------------------------------------------------------------------------------

def shl(x, n):
    return lax.shift_left(x, jnp.asarray(n, x.dtype))


def shr(x, n):
    return lax.shift_right_logical(x, jnp.asarray(n, x.dtype))


def bswitch(idx, fns):
    """Run fns[k]() for k == idx (traced scalar); keys sorted ints. Binary cascade of whens."""
    keys = sorted(fns)

    def rec(ks):
        if len(ks) == 1:
            fns[ks[0]]()
            return
        mid = len(ks) // 2

        @pl.when(idx < ks[mid])
        def _():
            rec(ks[:mid])

        @pl.when(idx >= ks[mid])
        def _():
            rec(ks[mid:])

    rec(keys)


def bf(x):
    """Round an f32 value to bf16 and back."""
    return x.astype(BF16).astype(F32)


def quant_rows(x):
    """Main's per-token dynamic fp8 activation quant (quantized_matmul/util.py:282-306
    quantize_block): abs_max (exact in bf16) / f32(448) promotes to f32, so the scale and x / scale
    are f32 (executed on jax 0.11.2, instruments/reference.py act_quant). -> (fp8, f32 [M, 1])."""
    s = jnp.max(jnp.abs(x), axis=1, keepdims=True) / FP8_MAX
    s = jnp.where(s == 0, 1.0, s)
    return jnp.clip(x / s, -FP8_MAX, FP8_MAX).astype(FP8), s


def quant_block_gmm(x):
    """gmm_v2 lhs quant of one block (gmm_v2.py:525-536): scale = bf16(abs_max / 448),
    inv = bf16(1 / scale) (0 if scale == 0), q = fp8(clip(bf16(x * inv))). -> (fp8, f32 [M, 1])."""
    s = bf(jnp.max(jnp.abs(x), axis=1, keepdims=True) / FP8_MAX)
    inv = jnp.where(s == 0, 0.0, bf(1.0 / jnp.where(s == 0, 1.0, s)))
    return jnp.clip(bf(x * inv), -FP8_MAX, FP8_MAX).astype(FP8), s


def rmsnorm(x, w, eps):
    """x f32 [M, H] (bf16-valued residual); w f32 [1, H] (bf16-valued). vLLM ir default: f32 norm ->
    bf16 -> x bf16 weight in bf16."""
    y = bf(x * lax.rsqrt(jnp.mean(x * x, axis=1, keepdims=True) + eps))
    return bf(y * w)


def lane_swap(x):
    """Interleaved-rope partner: out[2i] = x[2i+1], out[2i+1] = x[2i]."""
    n = x.shape[-1]
    lane = lax.broadcasted_iota(I32, x.shape, x.ndim - 1)
    return jnp.where(lane % 2 == 0, pltpu.roll(x, n - 1, x.ndim - 1), pltpu.roll(x, 1, x.ndim - 1))


def rope(x, cos, sin_signed):
    """GPT-J interleaved rope with torch bf16 elementwise rounding (each product and the sum rounded,
    common.py:162-185): bf16(bf16(x*cos) + bf16(swap(x)*sin)); cos/sin_signed [M, 128] bf16-valued
    (identity lanes carry cos=1, sin=0)."""
    return bf(bf(x * cos) + bf(lane_swap(x) * sin_signed))


def fp8_bytes(x8):
    return lax.bitcast_convert_type(x8, U8).astype(I32)


def fp8_decode_hw(b):
    """int32 byte (0..255, e4m3fn bits) -> bf16 via the hardware convert; NaN codes (0x7F / 0xFF,
    never written: writes clip to +-448) -> +-448."""
    b = jnp.where((b & 0x7F) == 0x7F, b ^ 1, b)
    return lax.bitcast_convert_type(b.astype(U8), FP8).astype(BF16)


def pack2(lo, hi):
    """Two bf16-valued f32 arrays -> one u32 array (lo's bf16 bits in the low half, hi's in the high
    half): exchanges of bf16 rows in 32-bit (8-row) tiles instead of bf16's 16-row tiles."""
    a = shr(lax.bitcast_convert_type(lo, U32), 16)
    b = lax.bitcast_convert_type(hi, U32) & jnp.uint32(0xFFFF0000)
    return a | b


def unpack_lo(u):
    return lax.bitcast_convert_type(shl(u, 16), F32)


def unpack_hi(u):
    return lax.bitcast_convert_type(u & jnp.uint32(0xFFFF0000), F32)


def mono_key(score):
    """f32 score -> order-preserving i32 key (sparsecore_topk: bits ^ ((bits >> 31) & 0x7FFFFFFF));
    -0.0 < +0.0."""
    b = lax.bitcast_convert_type(score, I32)
    return jnp.where(b >= 0, b, b ^ jnp.int32(0x7FFFFFFF))


def mxu(a, b, tiles):
    if tiles.dot == "fp8" or (a.dtype != FP8 and b.dtype != FP8):
        return a, b
    if tiles.dot == "mixed":
        return a.astype(BF16), b
    return a.astype(BF16), b.astype(BF16)


class Phases:
    def __init__(self, on):
        self.on, self.cur = on, None

    def mark(self, name):
        self.end()
        if self.on:
            self.cur = jax.named_scope(name)
            self.cur.__enter__()

    def end(self):
        if self.cur is not None:
            self.cur.__exit__(None, None, None)
            self.cur = None


def cache_groups(v, ck):
    """build()'s stack_c -> {kind: n_groups} for the stacked cache kinds. A set / bool stacks each
    kind as one operand; a dict {kind: n} splits a kind into n equal groups of consecutive layers
    (one operand each): SparseCore takes at most 16 GiB per tensor, so the runner stacks the full
    78-layer nope / rope caches in 2 groups (GLM5_MKD_STACK_C, runner.glm5_kv_stack_groups)."""
    if isinstance(v, dict):
        g = {k: int(n) for k, n in v.items() if int(n) > 0}
    else:
        g = {k: 1 for k in stack_kinds(v, ck)}
    for k, n in g.items():
        assert k in ck and len(ck[k][0]) % n == 0, (k, n)
    return g


def cache_names(ck, G):
    return [x for k, (ls, _, _) in ck.items()
            for x in ([(k, "stk")] if G.get(k) == 1 else
                      [(k, f"stk{i}") for i in range(G[k])] if k in G else [(k, l) for l in ls])]


def stack_kinds(v, kinds):
    if v is None:
        v = ()
    if isinstance(v, bool):
        v = kinds if v else ()
    v = frozenset(v)
    assert v <= set(kinds), sorted(v - set(kinds))
    return v


# ------------------------------------------------------------------------------------------
# The kernel
# ------------------------------------------------------------------------------------------

def build(cfg, *, nb, max_pages, bmax=None, tiles=Tiles(), interpret=False, debug=False,
          skip=frozenset(), axis_name=None, stack_w=None, stack_c=None, draft=False, dbg_part=True):
    """Returns (step_fn, info). step_fn(scal, rmeta, qmeta, bt, rvec, h0, rope_tab, smalls, *weights,
    *caches) -> (out [M, H] bf16, *caches[, dbg_h, dbg_own]) for ONE rank; wrap in shard_map.

    scal  SMEM i32 [8]:          0 live rows T, 1 requests B
    rmeta SMEM i32 [TMAX, 8]:    per row: 0 valid, 1 pos, 2 req, 3 seq_len (request, incl. this step)
    qmeta SMEM i32 [BMAX, 4]:    per request: 0 first row, 1 rows, 2 seq_len, 3 valid
    bt    SMEM i32 [BMAX, MAXP]: per request local page ids (DCP virtual page id; local page = id)
    rvec  VMEM i32 [M, 128]:     rmeta as vectors (lane 0 valid, 1 pos)
    h0    VMEM f32 [TMAX, H]:    embedding rows (bf16 values; f32 so a traced single-row load is legal)
    rope_tab VMEM f32 [4, M, 128]: per row cos / signed sin with rope in lanes 0..DR-1 (0, 1) and in
                                 lanes 128-DR..127 (2, 3); identity elsewhere
    smalls HBM f32 [L + 1, 1, S]: this rank's per-layer vectors (smalls_layout; row L: final norm)
    caches: aliased in -> out; new rows are written in place before they are read.
    skip: timing ablations only (numerics garbage).

    draft=True (cfg.V > 0; WHITEBOARD.v4): the MTP drafter -- L = 3 passes of the one MTP layer
    (weights stacked 3x; caches = the draft layer's, 3 views). scal / rmeta / qmeta / rvec / rope_tab
    gain a leading [3] (per pass) flattened into the row axis (rmeta [3*TMAX, 8], qmeta [3*BMAX, 4],
    rvec [3*M, 128], rope_tab [12, M, 128]); h0 = the target's post-final-norm hidden; after smalls come dmeta
    SMEM i32 [4, 128] (rows 0-2: per pass the rows whose logits are taken; row 3: pass 0's token ids),
    the vocab shard of embed [VR, H/2] (u32 pairs: bf16 row[:H/2] | row[H/2:] << 16) and lm_head
    [VRP, H] (bf16, HBM) and eh_proj^T's output-column
    shard [2, H/R, H] (bf16; 0: embed half, 1: hidden half). out = tokens [M, 128] i32 (row b, lane
    pass). Only pass 0 writes KV (passes 1-2's rows are dead stores in main, never attended)."""
    assert set(skip) <= set(ABLATIONS), skip
    assert not draft or cfg.V > 0
    R, L, H = cfg.R, cfg.L, cfg.H
    TMAX, RPR = cfg.TMAX, cfg.RPR
    BMAX = bmax or TMAX
    M = max(32, rup(TMAX, 8))
    QB8 = 8                                              # rows per request block (>= QMAX)
    assert cfg.QMAX <= QB8
    pg = cfg.page
    lg_pg = int(math.log2(pg))
    assert 1 << lg_pg == pg and pg % 128 == 0 and pg <= 256      # gather source packs page << 8 | slot
    NPR = pg // 128                                      # key rows per page
    KT = cfg.KT
    KR_ = KT // 128                                      # candidate rows
    GKR = rup(R * KR_, 128)                              # gathered candidate rows (padded)
    ROWSL = rup(max(1, cfg.n_local(max_pages * pg * R, 0)) // 128 + 1, 128)   # local key rows
    ROWSL = max(ROWSL, 128)
    assert ROWSL <= 128 * 1024
    UR_ = TMAX * KR_ + BMAX                              # flat union rows (sum of ceil(nu/128))
    lay, S = smalls_layout(cfg)
    wk = weight_kinds(cfg)
    ck = cache_kinds(cfg, nb)
    HR, NH, KL, DQ, DN, DV = cfg.HR, cfg.NH, cfg.KL, cfg.DQ, cfg.DN, cfg.DV
    NQB = HR * DQ
    QA = KL + 128                                        # q row per head: absorbed 512 | rope 64 | 0
    QH = HR * QA // 2                                    # packed q row (u32 pairs)
    OH = HR * KL // 2                                    # packed partial row (u32 pairs)
    OW = OH + 128                                        # + lse lanes (f32 bits, lane = local head)
    assert (HR * QA) % 256 == 0 and (HR * KL) % 256 == 0
    KRO = cfg.KR
    RO8 = max(8, rup(RPR, 8))
    NBLK13, NBLK2 = H // cfg.QB, cfg.I // cfg.QB
    assert H % cfg.QB == 0 and cfg.I % cfg.QB == 0 and cfg.QB % 128 == 0
    sm_scale = DQ ** -0.5
    IR4 = cfg.IREC // 4
    ILV, G_ = cfg.ILV, cfg.ILV * R

    def kdim(kind):
        return {"ew13": H, "ew2": cfg.I}.get(kind) or wk[kind][1][0]

    tk = {}
    for kind in ("qkv_a", "q_b", "o_proj", "idx_wq", "sw13", "sw2", "w13", "w2", "idx_wk", "gate"):
        K, N = wk[kind][1]
        tk[kind] = tiles.tk(K, N, wk[kind][2])
    tk["ew13"] = tiles.tk(H, 2 * cfg.I, FP8)
    tk["ew2"] = tiles.tk(cfg.I, H, FP8)
    for k_ in ("ew13", "ew2"):
        assert tk[k_] % 128 == 0

    NACC = rup(max(rup(cfg.QKVA), NQB, H, 2 * cfg.IDR, 2 * cfg.I, 2 * cfg.ISR, cfg.IH * cfg.IDIM,
                   rup(cfg.IDIM + cfg.IH), cfg.E))
    KX = max(H, cfg.QL, KRO, cfg.IDR, cfg.I, cfg.ISR)

    SW = frozenset(k for k in stack_kinds(stack_w, wk) if wk[k][0])   # never a 0-length stack (Mosaic SIGILL)
    SCG = cache_groups(stack_c, ck)
    SC = frozenset(SCG)
    w_names = [x for k, (ls, _, _) in wk.items()
               for x in ([(k, "stk")] if k in SW else [(k, l) for l in ls])]
    c_names = cache_names(ck, SCG)

    def lpos_fn(ls):
        ls = list(ls)
        n = len(ls)
        if not n:                                        # a kind no layer has (drafter: dense MLP)
            return lambda l: jnp.int32(0)
        if ls == list(range(ls[0], ls[0] + n)):
            a = ls[0]
            f = (lambda l: l) if a == 0 else (lambda l: l - a)
        else:
            fr, sk = cfg.idx_freq, cfg.idx_skip
            assert ls == [l for l in range(L) if cfg.full_indexer(l)], ls
            f = lambda l: jnp.where(l < sk, l, sk + (l - sk) // fr)
        for i, l in enumerate(ls):
            assert int(f(jnp.int32(l))) == i, (ls, l)
        return lambda l: jnp.clip(f(l), 0, n - 1)
    WPOS = {k: lpos_fn(ls) for k, (ls, _, _) in wk.items()}
    CPOS = {k: lpos_fn(ls) for k, (ls, _, _) in ck.items()}
    n_fixed = 12 if draft else 8
    n_w = len(w_names)
    if draft:
        VR = cfg.V // R                                  # this rank's vocab rows
        VRP = rup(VR, 128)
        XR = rup(BMAX, 8)                                # rows whose logits are taken
        DH = H // R                                      # eh_proj output columns per rank
        assert cfg.V % R == 0 and DH % 128 == 0 and XR <= M
    n_c = len(c_names)

    # --- scratch ---
    scratch = {
        "sm": pltpu.VMEM((2, 1, S), F32), "sm_sem": pltpu.SemaphoreType.DMA((2,)),
        # owner rows
        "h": pltpu.VMEM((RO8, H), F32), "res": pltpu.VMEM((RO8, H), F32),
        "hsend": pltpu.VMEM((RPR, H), BF16), "xall": pltpu.VMEM((R, RPR, H), BF16),
        # all rows
        "xbf": pltpu.VMEM((M, H), BF16), "xq": pltpu.VMEM((M, KX), FP8),
        "acc": pltpu.VMEM((M, NACC), F32),
        "qc": pltpu.VMEM((M, cfg.QL), F32), "kvc": pltpu.VMEM((M, KL), F32),
        "kpe": pltpu.VMEM((M, 128), F32),
        # KV / idx writes
        "wrow": pltpu.VMEM((TMAX, 128), U32), "rwrow": pltpu.VMEM((TMAX, 128), U32),
        "rec_v": pltpu.VMEM((TMAX, 256), I32),
        "rec_s": pltpu.SMEM((TMAX, 256), I32), "kvw_sem": pltpu.SemaphoreType.DMA((3,)),
        # q head-TP + AG
        "qsend": pltpu.VMEM((TMAX, QH), U32), "qall": pltpu.VMEM((R, TMAX, QH), U32),
        # indexer
        "iqv": pltpu.VMEM((cfg.IH, M + QB8, 128), F32), "wts": pltpu.VMEM((M + QB8, 128), F32),
        "ikr": pltpu.VMEM((M, 128), F32),
        "ipage": pltpu.VMEM((tiles.scan_depth, IR4, 4, pg), U8),
        "ip_sem": pltpu.SemaphoreType.DMA((tiles.scan_depth,)),
        "keys": pltpu.VMEM((cfg.QMAX, ROWSL, 128), I32),
        "kpre": pltpu.VMEM((max(ROWSL, 128), 128), I32),
        "tk_acc": pltpu.VMEM((min(max(tiles.topk_kc, tiles.topk_kc_global), KT), 768), F32),
        "tk_cc": pltpu.VMEM((2, 128, 128), F32),
        "cidx": pltpu.VMEM((KR_, 128), I32),
        "ksend": pltpu.VMEM((TMAX, KR_, 128), I32), "kall": pltpu.VMEM((R, TMAX, KR_, 128), I32),
        "gk": pltpu.VMEM((GKR, 128), I32), "gmk": pltpu.VMEM((128, 128), I32),
        "oslot": pltpu.VMEM((KR_, 128), I32),
        "cand_s": pltpu.SMEM((TMAX * KR_, 128), I32), "oslot_s": pltpu.SMEM((KR_, 128), I32),
        "cand_v": pltpu.VMEM((TMAX, KR_, 128), I32),
        "own_s": pltpu.SMEM((TMAX * KR_, 128), I32), "own_v": pltpu.VMEM((TMAX, KR_, 128), I32),
        "nown_s": pltpu.SMEM((TMAX,), I32),
        # per-request union of the rows' owned winners (+ per-row membership over it)
        "own_u": pltpu.SMEM((UR_, 128), I32), "own_uv": pltpu.VMEM((UR_, 128), I32),
        "memb": pltpu.SMEM((UR_, 128), I32), "memb_v": pltpu.VMEM((UR_, 128), I32),
        "nu_s": pltpu.SMEM((BMAX,), I32), "uo_s": pltpu.SMEM((BMAX,), I32), "nuo_s": pltpu.SMEM((BMAX,), I32),
        "misc_sem": pltpu.SemaphoreType.DMA(()),
        # attention
        "g_nope": pltpu.VMEM((2, KT, 128), U32), "g_rope": pltpu.VMEM((2, KT, 128), U32),
        "g_sem": pltpu.SemaphoreType.DMA((2, 2)),
        "kvb": pltpu.VMEM((KT, KL), BF16),
        # a request's rows stacked on M: q (nope, rope by segment), online-softmax state
        "qb": pltpu.VMEM((cfg.QMAX * NH, KL), BF16),
        "q0b": pltpu.VMEM((4, cfg.QMAX * NH, 128), BF16), "q1b": pltpu.VMEM((4, cfg.QMAX * NH, 128), BF16),
        "att_acc": pltpu.VMEM((cfg.QMAX * NH, KL), F32),
        "m_s": pltpu.VMEM((cfg.QMAX * NH, 1), F32), "l_s": pltpu.VMEM((cfg.QMAX * NH, 1), F32),
        "osend": pltpu.VMEM((R, TMAX, OW), U32), "orecv": pltpu.VMEM((R, TMAX, OW), U32),
        "orow": pltpu.VMEM((M, KRO), F32),
        # reduce-scatter
        "part": pltpu.VMEM((M, H), F32),
        "psend": pltpu.VMEM((R, RPR, H), BF16), "rsrecv": pltpu.VMEM((R, RPR, H), BF16),
        # MoE
        "wloc": pltpu.VMEM((M, 128), F32), "routed": pltpu.VMEM((M, H), F32),
        "shared": pltpu.VMEM((M, H), F32),
        "sxb": pltpu.VMEM((NBLK13, M, 128), F32), "sxb2": pltpu.VMEM((NBLK2, M, 128), F32), "xqm": pltpu.VMEM((M, H), FP8),
        "hitv": pltpu.VMEM((8, 128), I32), "hits": pltpu.SMEM((8, 128), I32),
        "es13": pltpu.VMEM((cfg.ER + cfg.KREP, 2 * cfg.I), F32),
        "es2": pltpu.VMEM((cfg.ER + cfg.KREP, H), F32),
        "es_sem": pltpu.SemaphoreType.DMA((2,)),
    }
    for nm in ("ag", "q", "k", "a2a", "rso", "rsm"):
        scratch[nm + "_send"] = pltpu.SemaphoreType.DMA(())
        scratch[nm + "_recv"] = pltpu.SemaphoreType.DMA(())
    wshape = {
        "qkv_a": (tk["qkv_a"], rup(cfg.QKVA)), "q_b": (tk["q_b"], NQB),
        "o_proj": (tk["o_proj"], H), "idx_wq": (tk["idx_wq"], cfg.IH * cfg.IDIM),
        "idx_wk": (tk["idx_wk"], rup(cfg.IDIM + cfg.IH)), "gate": (tk["gate"], cfg.E),
        "sw13": (tk["sw13"], 2 * cfg.ISR), "sw2": (tk["sw2"], H),
        "ew13": (tk["ew13"], 2 * cfg.I), "ew2": (tk["ew2"], H),
        "w13": (tk["w13"], 2 * cfg.IDR), "w2": (tk["w2"], H),
        "w_uk": (HR, DN, KL), "w_uv": (HR, KL, DV),
    }
    wdtype = {k: (BF16 if k in ("idx_wk", "gate", "w_uk", "w_uv") else FP8) for k in wshape}
    NB = tiles.nbank
    BROWS = tiles.bank // 128
    for k, shp in wshape.items():
        assert pool_alias.storage_size(shp, jnp.dtype(wdtype[k])) <= tiles.bank, (k, shp)
    typed = interpret or os.environ.get("MKD_TYPED") == "1"
    tkinds = [k for k, shp in wshape.items() if typed or shp[-1] % 128]
    for k in tkinds:
        scratch["ibuf_" + k] = pltpu.VMEM((NB,) + wshape[k], wdtype[k])
    if len(tkinds) < len(wshape):
        scratch["pool"] = pltpu.VMEM((NB * BROWS, 128), U8)
    scratch["pool_sem"] = pltpu.SemaphoreType.DMA((NB,))
    scratch["st"] = pltpu.SMEM((8,), I32)
    if draft:
        scratch["lmb"] = pltpu.VMEM((128, H), BF16)      # lm_head / eh_proj tiles
        scratch["demb"] = pltpu.VMEM((1, M, H // 2), U32)   # embed rows (u32 pairs; 3-D: row DMAs)
        scratch["lm_sem"] = pltpu.SemaphoreType.DMA(())
        scratch["carry"] = pltpu.VMEM((XR, H), F32)      # next pass's previous hidden rows
        scratch["dsend"] = pltpu.VMEM((XR, 128), I32)
        scratch["drecv"] = pltpu.VMEM((R, XR, 128), I32)
        scratch["tokv"] = pltpu.VMEM((XR, 128), I32)
        scratch["tok_s"] = pltpu.SMEM((XR, 128), I32)
        scratch["d_send"] = pltpu.SemaphoreType.DMA(())
        # this pass's metadata, copied in the target's operand shapes
        scratch["scal_s"] = pltpu.SMEM((8,), I32)
        scratch["rmeta_s"] = pltpu.SMEM((TMAX, 8), I32)
        scratch["qmeta_s"] = pltpu.SMEM((BMAX, 4), I32)
        scratch["rvec_s"] = pltpu.VMEM((M, 128), I32)
        scratch["tab_s"] = pltpu.VMEM((4, M, 128), F32)
        scratch["d_recv"] = pltpu.SemaphoreType.DMA(())
    if debug:
        scratch["dbg_sem"] = pltpu.SemaphoreType.DMA((3,))
        scratch["dbgl"] = pltpu.VMEM((TMAX, 128), F32)
        if dbg_part and not draft:               # 2 MiB: the drafter's / adapter's debug builds go without
            scratch["dbgo"] = pltpu.VMEM((TMAX, NH, KL), F32)

    # --- the stream: one layer's tiles in consumption order ---
    SEGS = ("qkv_a", "q_b", "w_uk", "idx_wq", "idx_wk", "w_uv", "o_proj", "w13", "w2",
            "sw13", "gate", "sw2", "exp")
    SID = {k: i for i, k in enumerate(SEGS)}
    EXP = SID["exp"]
    KSEG = {k: (EXP if k in ("ew13", "ew2") else SID[k]) for k in wshape}

    def ntiles(kind):
        if kind in ("w_uk", "w_uv"):
            return 1
        return kdim(kind) // tk[kind]
    NTS = {k: ntiles(k) for k in SEGS if k != "exp"}
    T13, T2 = ntiles("ew13"), ntiles("ew2")
    TE = T13 + T2

    def bits(ks):
        return sum(1 << SID[k] for k in ks)
    M_BASE = bits(["qkv_a", "q_b", "w_uk", "w_uv", "o_proj"])
    M_IDX = 0 if "index" in skip else bits(["idx_wq", "idx_wk"])
    M_DENSE, M_MOE = bits(["w13", "w2"]), bits(["sw13", "sw2", "gate", "exp"])

    def segs_py(l):
        m = M_BASE | (M_IDX if cfg.full_indexer(l) else 0) | (M_MOE if cfg.is_moe(l) else M_DENSE)
        return [k for k in SEGS if m >> SID[k] & 1]

    reach = {k: {SID[k]} for k in SEGS}
    reach["init"] = set()
    for nh in sorted({0, 1, cfg.ER, cfg.ER + cfg.KREP}):
        seq = [SID[k] for l in range(L) for k in segs_py(l)
               for _ in range(nh * TE if k == "exp" else NTS[k])]
        reach["init"] |= set(seq[:NB])
        for p_, k in enumerate(seq):
            reach[SEGS[k]] |= set(seq[p_ + 1:p_ + 1 + NB])
    s_names = list(scratch)

    def kernel(*refs):
        if draft:
            scal_a, rmeta_a, qmeta_a, bt, rvec_a, h0, tab_a, smalls, dmeta, emb_h, lm_h, eht_h = refs[:n_fixed]
        else:
            scal, rmeta, qmeta, bt, rvec, h0, rope_tab, smalls = refs[:n_fixed]
        w_in = refs[n_fixed:n_fixed + n_w]
        c_in = refs[n_fixed + n_w:n_fixed + n_w + n_c]          # read-only (see build_writer)
        o0 = n_fixed + n_w + n_c
        out_ref, nrow_out, rrow_out, irec_out = refs[o0:o0 + 4]
        rest = refs[o0 + 4:]
        if debug:
            dbg_h, dbg_own, dbg_lse, dbg_part = rest[:4]
            rest = rest[4:]
        scr = dict(zip(s_names, rest))
        if draft:
            scal, rmeta, qmeta = scr["scal_s"], scr["rmeta_s"], scr["qmeta_s"]
            rvec, rope_tab = scr["rvec_s"], scr["tab_s"]

            def load_pass(p_):
                for i_ in range(8):
                    scal[i_] = scal_a[p_, i_]

                def cr(t_, c):
                    for j_ in range(8):
                        rmeta[t_, j_] = rmeta_a[p_ * TMAX + t_, j_]
                    return c
                lax.fori_loop(0, TMAX, cr, 0)

                def cq(b_, c):
                    for j_ in range(4):
                        qmeta[b_, j_] = qmeta_a[p_ * BMAX + b_, j_]
                    return c
                lax.fori_loop(0, BMAX, cq, 0)
                r0_ = pl.multiple_of(p_ * M, 8)
                rvec[...] = rvec_a[pl.ds(r0_, M), :]
                for i_ in range(4):
                    rope_tab[i_] = tab_a[p_ * 4 + i_]
            load_pass(0)
        W, C, WS, CS = {}, {}, {}, {}
        for (k, l), r in zip(w_names, w_in):
            if l == "stk":
                WS[k] = r
                W[k] = {ll: r.at[i] for i, ll in enumerate(wk[k][0])}
            else:
                W.setdefault(k, {})[l] = r
        for (k, l), r in zip(c_names, c_in):
            if isinstance(l, str):                               # a stacked group
                CS.setdefault(k, []).append(r)
            else:
                C.setdefault(k, {})[l] = r
        for k, refs in CS.items():                               # static per-layer views (waits)
            gs = len(ck[k][0]) // len(refs)
            C[k] = {ll: refs[i // gs].at[i % gs] for i, ll in enumerate(ck[k][0])}

        def wsel(kind, l, fn):
            if kind in SW:
                fn(WS[kind].at[WPOS[kind](l)])
            else:
                bswitch(l, {ll: (lambda r=r: fn(r)) for ll, r in W[kind].items()})

        def csel(kinds, l, fn):
            if all(k in SC for k in kinds):
                ng = len(CS[kinds[0]])
                assert all(len(CS[k]) == ng for k in kinds), kinds
                pos = CPOS[kinds[0]](l)
                if ng == 1:
                    fn(*[CS[k][0].at[pos] for k in kinds])
                else:
                    gs = len(ck[kinds[0]][0]) // ng
                    bswitch(pos // gs, {gi: (lambda gi=gi: fn(*[CS[k][gi].at[pos - gi * gs]
                                                                 for k in kinds]))
                                        for gi in range(ng)})
            else:
                ks = sorted(C[kinds[0]])
                bswitch(l, {ll: (lambda ll=ll: fn(*[C[k][ll] for k in kinds])) for ll in ks})

        me = scal[7] if axis_name is None else lax.axis_index(axis_name).astype(I32)
        T = scal[0]
        B = scal[1]
        nrows = jnp.minimum((T + 7) // 8 * 8, TMAX)             # exchanged rows (32-bit tiles)
        lane128 = lax.broadcasted_iota(I32, (1, 128), 1)
        row_iota = lax.broadcasted_iota(I32, (M, 1), 0)
        rvalid = (rvec[:, 0:1] > 0).astype(F32)                 # [M, 1]
        rpos = rvec[:, 1:2]                                      # [M, 1] i32

        # ---------------- weight streaming (one ring, one producer) ----------------
        st, psem = scr["st"], scr["pool_sem"]

        def src_of(kind, ref, t, j=None):
            if kind in ("w_uk", "w_uv"):
                return ref
            if kind in ("ew13", "ew2"):
                return ref.at[j, pl.ds(t * tk[kind], tk[kind])]
            return ref.at[pl.ds(t * tk[kind], tk[kind])]

        def view(kind, slot):
            if kind in tkinds:
                return scr["ibuf_" + kind].at[slot]
            shp, dt = wshape[kind], jnp.dtype(wdtype[kind])
            return pool_alias.view(
                scr["pool"].at[pl.ds(slot * BROWS, pool_alias.storage_size(shp, dt) // 128)], shp, dt)

        def copy(kind, src, slot):
            return pltpu.make_async_copy(src, view(kind, slot), psem.at[slot])

        def seg_mask(l):
            full = jnp.logical_or(l < cfg.idx_skip, (l - cfg.idx_skip) % cfg.idx_freq == cfg.idx_freq - 1)
            return (jnp.int32(M_BASE) | jnp.where(full, jnp.int32(M_IDX), 0)
                    | jnp.where(l >= cfg.first_dense, jnp.int32(M_MOE), jnp.int32(M_DENSE)))

        def nt_dyn(sg):
            v = st[6] * TE
            for k, n in NTS.items():
                v = jnp.where(sg == SID[k], n, v)
            return v

        def issue_seg(a, l, t, slot):
            kind = SEGS[a]
            if kind != "exp":
                wsel(kind, l, lambda r: copy(kind, src_of(kind, r, t), slot).start())
                return
            i = t // TE
            u = t - i * TE
            j = scr["hits"][1, i]

            def exp_tiles(k13, k2, jj):
                @pl.when(u < T13)
                def _():
                    wsel(k13, l, lambda r: copy("ew13", src_of("ew13", r, u, jj), slot).start())

                @pl.when(u >= T13)
                def _():
                    wsel(k2, l, lambda r: copy("ew2", src_of("ew2", r, u - T13, jj), slot).start())
            if not cfg.KREP:
                exp_tiles("ew13", "ew2", j)
                return

            @pl.when(j < cfg.ER)
            def _():
                exp_tiles("ew13", "ew2", j)

            @pl.when(j >= cfg.ER)                    # replica slot q = j - ER
            def _():
                exp_tiles("rw13", "rw2", j - cfg.ER)

        def pump(site, allowed=None):
            """Issue stream tiles while a bank is free. st: 0 issued, 1 consumed, 2 layer, 3 segment,
            4 tile in segment, 5 layer whose hit list is published, 6 its hit count."""
            if "weights" in skip:
                return
            allowed = sorted(reach[site] if allowed is None else allowed)
            amask = jnp.int32(sum(1 << a for a in allowed))

            def cond(c):
                l, sg = st[2], st[3]
                ok = jnp.logical_and(st[0] - st[1] < NB, l < L)
                ok = jnp.logical_and(ok, (lax.shift_right_logical(amask, sg) & 1) == 1)
                return jnp.logical_and(ok, jnp.logical_or(sg != EXP, st[5] == l))

            def body(c):
                l, sg, t = st[2], st[3], st[4]
                nt = nt_dyn(sg)

                @pl.when(t < nt)
                def _():
                    slot = st[0] % NB
                    bswitch(sg, {a: (lambda a=a: issue_seg(a, l, t, slot)) for a in allowed})
                    st[0] = st[0] + 1
                t1 = jnp.where(t < nt, t + 1, t)
                st[4] = t1

                @pl.when(t1 >= nt)
                def _():
                    rest_ = lax.shift_right_logical(seg_mask(l), sg + 1)
                    low = rest_ & (-rest_)
                    has = rest_ != 0
                    st[3] = jnp.where(has, sg + 1 + (jnp.int32(31) - lax.clz(low)), 0)
                    st[2] = jnp.where(has, l, l + 1)
                    st[4] = 0
                return c + 1
            lax.while_loop(cond, body, jnp.int32(0))

        def take(kind):
            if "weights" in skip:
                return view(kind, 0)

            @pl.when(st[0] <= st[1])
            def _():
                pump(kind if kind in SID else "exp", allowed={KSEG[kind]})
            slot = st[1] % NB
            ref0 = next(iter(W[kind].values()))
            copy(kind, src_of(kind, ref0, 0, 0), slot).wait()
            return view(kind, slot)

        def give(kind):
            st[1] = st[1] + 1
            pump(kind if kind in SID else "exp")

        def matmul(kind, l, xref, K, N, j=None):
            """acc[:, :N] = xref[:, :K] @ W, W's tiles taken off the stream in order."""
            nt = ntiles(kind)
            t_k = tk[kind]
            acc = scr["acc"]
            acc[:, :N] = jnp.zeros((M, N), F32)

            def body(t, c):
                w_ = take(kind)
                k0 = pl.multiple_of(t * t_k, t_k)
                xs, ws = mxu(xref[:, pl.ds(k0, t_k)], w_[...], tiles)
                acc[:, :N] += lax.dot_general(xs, ws, NN, preferred_element_type=F32)
                give(kind)
                return c
            lax.fori_loop(0, nt, body, 0)
            return acc[:, :N]

        def matmul_blk(kind, l, xref, K, N, j, sname):
            """Expert GEMM with gmm_v2's lhs quant per row per QB-column block: acc[:, :N] =
            sum over 128-row segments s of (xq[:, s] @ W[s]) * sx[block(s)] (f32; main accumulates the
            blocks in bf16 -- deliberate, WHITEBOARD §10). sxb[b] holds block b's row scales."""
            nt = ntiles(kind)
            t_k = tk[kind]
            acc = scr["acc"]
            acc[:, :N] = jnp.zeros((M, N), F32)

            def body(t, c):
                w_ = take(kind)
                for sg in range(t_k // 128):
                    k0 = t * t_k + sg * 128
                    blk = k0 // cfg.QB
                    xs, ws = mxu(xref[:, pl.ds(pl.multiple_of(k0, 128), 128)],
                                 w_[sg * 128:(sg + 1) * 128, :], tiles)
                    acc[:, :N] += (lax.dot_general(xs, ws, NN, preferred_element_type=F32)
                                   * scr[sname][blk][:, 0:1])
                give(kind)
                return c
            lax.fori_loop(0, nt, body, 0)
            return acc[:, :N]

        def blk_quant(x, nblk, sname):
            """x f32 [M, nblk*QB] bf16-valued -> xq fp8 (returned) and scr[sname][b] = block b's
            row scales (separate buffers for GEMM1 / GEMM2: GEMM2's quant runs between experts)."""
            out = []
            for b in range(nblk):
                xb = x[:, b * cfg.QB:(b + 1) * cfg.QB]
                q, s = quant_block_gmm(xb)
                scr[sname][b] = jnp.broadcast_to(s, (M, 128))
                out.append(q)
            return jnp.concatenate(out, axis=1)

        def sm(slot, name):
            o, n = lay[name]
            return scr["sm"][slot, :, o:o + n]

        # the stream starts before anything else
        for i_, v_ in enumerate((0, 0, 0, 0, 0, -1, 0, 0)):
            st[i_] = jnp.int32(v_)
        pump("init")

        # ---------------- collectives ----------------
        if R > 1:
            barrier = pltpu.get_barrier_semaphore()
            for d in range(R):
                pl.semaphore_signal(barrier, inc=1, device_id=(jnp.int32(d),),
                                    device_id_type=pl.DeviceIdType.MESH)
            pl.semaphore_wait(barrier, R)

        def exchange(src_of_d, dst, ss, rs, wait=True, rows=None):
            """Start src_of_d(d) -> dst[me] on every peer d; wait R-1 receives and sends.
            Race freedom (WHITEBOARD §5): every exchange is all-to-all, and each receive buffer is
            consumed before this rank starts the next exchange whose completion peers need before
            they rewrite it."""
            if R == 1 or "coll" in skip:
                return None if wait else (lambda: None)
            src_fn = src_of_d if rows is None else (lambda d: src_of_d(d).at[pl.ds(0, rows)])
            dst_fn = (lambda i: dst.at[i]) if rows is None else (lambda i: dst.at[i, pl.ds(0, rows)])
            for d in range(R):
                @pl.when(me != d)
                def _(d=d):
                    pltpu.make_async_remote_copy(
                        src_ref=src_fn(d), dst_ref=dst_fn(me), send_sem=scr[ss], recv_sem=scr[rs],
                        device_id=(jnp.int32(d),), device_id_type=pl.DeviceIdType.MESH).start()

            def finish():
                desc = pltpu.make_async_remote_copy(
                    src_ref=src_fn(0), dst_ref=dst_fn(0), send_sem=scr[ss], recv_sem=scr[rs],
                    device_id=(me,), device_id_type=pl.DeviceIdType.MESH)
                for _ in range(R - 1):
                    desc.wait_recv()
                for _ in range(R - 1):
                    desc.wait_send()
            if not wait:
                return finish
            finish()

        def gather_rows_all(x_own_f32, ss, rs):
            """Owner rows (RO8, H) f32 (rows < RPR live) -> X [M, H] f32, all rows in t order."""
            scr["hsend"][...] = x_own_f32[0:RPR].astype(BF16)
            scr["xall"][me] = scr["hsend"][...]
            exchange(lambda d: scr["hsend"], scr["xall"], ss, rs)
            rows = [scr["xall"][s_, k_:k_ + 1, :] for k_ in range(RPR) for s_ in range(R)]
            if M > TMAX:
                rows.append(jnp.zeros((M - TMAX, H), BF16))
            return jnp.concatenate(rows, axis=0).astype(F32)

        def reduce_scatter_rows(part_bf16, ss, rs):
            """part [M, H] (bf16 values) -> sum over ranks of this rank's owned rows (RPR, H) f32."""
            for d in range(R):
                scr["psend"][d] = jnp.concatenate(
                    [part_bf16[d + R * k_:d + R * k_ + 1, :] for k_ in range(RPR)], axis=0).astype(BF16)
            scr["rsrecv"][me] = scr["psend"][me]
            exchange(lambda d: scr["psend"].at[d], scr["rsrecv"], ss, rs)
            a = scr["rsrecv"][0].astype(F32)
            for s_ in range(1, R):
                a = a + scr["rsrecv"][s_].astype(F32)
            return a                                                 # (RPR, H) f32

        def own_rows_pad(x):
            """(RPR, H) -> (RO8, H)."""
            if RO8 > RPR:
                return jnp.concatenate([x, jnp.zeros((RO8 - RPR, H), F32)], axis=0)
            return x

        def own_valid():
            v = []
            for k_ in range(RPR):
                t_ = me + R * k_
                v.append(jnp.where(jnp.logical_and(t_ < T, rmeta[jnp.minimum(t_, TMAX - 1), 0] > 0),
                                   1.0, 0.0))
            col = jnp.zeros((RO8, 1), F32)
            ri = lax.broadcasted_iota(I32, (RO8, 1), 0)
            for k_, vv in enumerate(v):
                col = jnp.where(ri == k_, vv, col)
            return col

        def ow_page(t_):
            """(page, slot) of row t_'s new KV on its owner."""
            pos = rmeta[t_, 1]
            req = rmeta[t_, 2]
            lo = (pos // G_) * ILV + pos % ILV
            return bt[req, lo >> lg_pg], lo & (pg - 1)

        def ow_mine(t_):
            pos = rmeta[t_, 1]
            return jnp.logical_and(jnp.logical_and(t_ < T, rmeta[t_, 0] > 0), (pos % G_) // ILV == me)

        # ---------------- new-row staging (the caches are read-only in this kernel) ----------------
        def stage_rows(l):
            """This layer's new KV words for all T rows: nope words -> wrow, the rope token's 32 words
            placed at its quarter (pos & 3) of the 4-token row -> rwrow. Both are copied to the nrow /
            rrow outputs (applied to the caches by build_writer after the call) and serve the gather
            patch for this step's own new rows."""
            nb8 = fp8_bytes(jnp.clip(scr["kvc"][0:TMAX], -FP8_MAX, FP8_MAX).astype(FP8))
            word = (nb8[:, 0:128] | shl(nb8[:, 128:256], 8) | shl(nb8[:, 256:384], 16)
                    | shl(nb8[:, 384:512], 24))
            scr["wrow"][...] = lax.bitcast_convert_type(word, U32)
            rb = fp8_bytes(jnp.clip(scr["kpe"][0:TMAX], -FP8_MAX, FP8_MAX).astype(FP8))
            lane_t = lax.broadcasted_iota(I32, (TMAX, 128), 1)
            rw = jnp.where(lane_t < 32, rb | shl(pltpu.roll(rb, 96, 1), 8), 0)   # w j = b[j] | b[32+j]<<8
            seg = rvec[0:TMAX, 1:2] & 3
            rq = jnp.zeros((TMAX, 128), I32)
            for sg in range(4):
                rq = jnp.where((seg == sg) & (lane_t // 32 == sg),
                               pltpu.roll(rw, 32 * sg, 1) if sg else rw, rq)
            scr["rwrow"][...] = lax.bitcast_convert_type(rq, U32)
            pltpu.make_async_copy(scr["wrow"], nrow_out.at[l], scr["kvw_sem"].at[0]).start()
            pltpu.make_async_copy(scr["rwrow"], rrow_out.at[l], scr["kvw_sem"].at[1]).start()

        def stage_wait(l):
            pltpu.make_async_copy(scr["wrow"], nrow_out.at[l], scr["kvw_sem"].at[0]).wait()
            pltpu.make_async_copy(scr["rwrow"], rrow_out.at[l], scr["kvw_sem"].at[1]).wait()

        def req_new(req, qm=None):
            """(first row, rows, first new position, this rank's local length before the step).
            qm: optional getter (req, field) -> value (the drafter's pass-0 geometry)."""
            qm = (lambda r_, k_: qmeta[r_, k_]) if qm is None else qm
            t0 = qm(req, 0)
            nq = qm(req, 1)
            p0 = qm(req, 2) - nq
            nold = (p0 // G_) * ILV + jnp.clip(p0 % G_ - ILV * me, 0, ILV)
            return t0, nq, p0, nold

        # ---------------- one layer ----------------
        def layer(l, carry):
            nonlocal T, B, nrows, rvalid, rpos
            if draft:                                    # this pass's rows
                load_pass(l)
                T, B = scal[0], scal[1]
                nrows = jnp.minimum((T + 7) // 8 * 8, TMAX)
                rvalid = (rvec[:, 0:1] > 0).astype(F32)
                rpos = rvec[:, 1:2]

                @pl.when(jnp.logical_and(l == 1, "d_ishare" not in skip))
                def _():
                    pd = Phases(tiles.trace)
                    pd.mark("D_ishare")
                    index_share()
                    pd.end()
            ph = Phases(tiles.trace)
            ph.mark("L_norm")
            slot = l % 2
            is_moe = l >= cfg.first_dense
            full_idx = jnp.logical_or(
                l < cfg.idx_skip,
                jnp.logical_and(l >= cfg.idx_skip,
                                (l - cfg.idx_skip) % cfg.idx_freq == cfg.idx_freq - 1))
            pltpu.make_async_copy(smalls.at[0], scr["sm"].at[slot], scr["sm_sem"].at[slot]).wait()

            @pl.when(l + 1 < L)
            def _():
                pltpu.make_async_copy(smalls.at[l + 1], scr["sm"].at[1 - slot],
                                      scr["sm_sem"].at[1 - slot]).start()

            # 1-2: owner input norm, AG normed rows
            x = scr["h"][...] + scr["res"][...]                          # f32 sum, unrounded
            scr["res"][...] = bf(x)
            hn = rmsnorm(x, sm(slot, "in_ln"), cfg.eps) * own_valid()
            X = gather_rows_all(hn, "ag_send", "ag_recv")                  # [M, H] f32
            scr["xbf"][...] = X.astype(BF16)

            # 3: replicated fused_qkv_a, norms, k_pe rope
            ph.mark("L_qkva")
            xq, sx = quant_rows(X)
            scr["xq"][:, :H] = xq
            a = matmul("qkv_a", l, scr["xq"], H, rup(cfg.QKVA))
            o_s, _ = lay["qkv_a_s"]
            qkv_s = scr["sm"][slot, :, o_s:o_s + rup(cfg.QKVA)]
            y = bf(a * sx * qkv_s)
            qc = rmsnorm(y[:, :cfg.QL], sm(slot, "q_a_ln"), cfg.eps) * rvalid
            kvc = rmsnorm(y[:, cfg.QL:cfg.QL + KL], sm(slot, "kv_a_ln"), cfg.eps) * rvalid
            o_pe = cfg.QL + KL
            kpe_raw = y[:, o_pe:o_pe + 128] if o_pe + 128 <= rup(cfg.QKVA) else jnp.concatenate(
                [y[:, o_pe:], jnp.zeros((M, o_pe + 128 - rup(cfg.QKVA)), F32)], axis=1)
            kpe_raw = jnp.where(lax.broadcasted_iota(I32, (M, 128), 1) < cfg.DR, kpe_raw, 0.0)
            kpe = rope(kpe_raw, rope_tab[0], rope_tab[1]) * rvalid
            scr["qc"][...] = qc
            scr["kvc"][...] = kvc
            scr["kpe"][...] = kpe

            # 5: q_b on this rank's heads, rope, W_UK absorb -> qsend
            ph.mark("L_qb")
            xq, sx = quant_rows(qc)
            scr["xq"][:, :cfg.QL] = xq
            a = matmul("q_b", l, scr["xq"], cfg.QL, NQB)
            o_qb, _ = lay["q_b_s"]
            q = bf(a * sx * scr["sm"][slot, :, o_qb:o_qb + NQB])            # [M, NQB]
            wb = take("w_uk")
            parts = []
            for hh in range(HR):
                qn = q[:, hh * DQ:hh * DQ + DN].astype(BF16)
                ql = bf(lax.dot_general(qn, wb[hh], NN, preferred_element_type=F32))   # [M, KL]
                win = q[:, hh * DQ + DQ - 128:hh * DQ + DQ]                  # nope tail | rope
                wr = rope(win, rope_tab[2], rope_tab[3])                     # rope lanes 128-DR..
                qpe = jnp.where(lax.broadcasted_iota(I32, (M, 128), 1) < cfg.DR,
                                pltpu.roll(wr, cfg.DR, 1), 0.0)              # -> lanes 0..DR-1
                parts += [ql[0:TMAX], qpe[0:TMAX]]
            give("w_uk")
            qrow = jnp.concatenate(parts, axis=1)                            # [TMAX, HR*QA] bf16 vals
            scr["qsend"][...] = pack2(qrow[:, :QH], qrow[:, QH:])
            scr["qall"][me] = scr["qsend"][...]
            q_wait = exchange(lambda d: scr["qsend"], scr["qall"], "q_send", "q_recv", wait=False,
                              rows=nrows)

            # 4: this layer's new KV rows -> staging (the AG of q flies meanwhile)
            ph.mark("L_kvw")
            if draft:
                pl.when(l == 0)(lambda: stage_rows(l))
            else:
                stage_rows(l)

            @pl.when(jnp.logical_not(full_idx))
            def _():
                attn_first(l)

            # 7: indexer (fresh top-k layers)
            ph.mark("L_index")

            @pl.when(jnp.logical_and(full_idx, "index" not in skip))
            def _():
                indexer(l, slot)

            @pl.when(full_idx)
            def _():
                attn_first(l)

            # 9: per row: gather + attention -> partial + lse
            ph.mark("L_attn")
            q_wait()
            attention(l)

            # 10: A2A partials to head owners, LSE merge
            ph.mark("L_merge")
            scr["orecv"][me] = scr["osend"][me]
            exchange(lambda d: scr["osend"].at[d], scr["orecv"], "a2a_send", "a2a_recv", rows=nrows)
            o_h = merge()                                                    # HR x [M, KL] f32

            # 11: W_UV + o_proj (local K, per-rank scale)
            ph.mark("L_oproj")
            wv = take("w_uv")
            for hh in range(HR):
                y_ = lax.dot_general(o_h[hh].astype(BF16), wv[hh], NN, preferred_element_type=F32)
                scr["orow"][:, hh * DV:(hh + 1) * DV] = bf(y_)
            give("w_uv")
            xq, sx = quant_rows(scr["orow"][...])
            scr["xq"][:, :KRO] = xq
            a = matmul("o_proj", l, scr["xq"], KRO, H)
            part = bf(a * sx * sm(slot, "o_proj_s"))
            ph.mark("L_rs_attn")
            attn = reduce_scatter_rows(part, "rso_send", "rso_recv")         # (RPR, H)
            ph.mark("L_ag_post")

            # 13: owner post norm, AG
            x = own_rows_pad(bf(attn)) + scr["res"][...]
            scr["res"][...] = bf(x)
            hn = rmsnorm(x, sm(slot, "post_ln"), cfg.eps) * own_valid()
            X = gather_rows_all(hn, "ag_send", "ag_recv")
            scr["xbf"][...] = X.astype(BF16)

            ph.mark("L_mlp")

            if cfg.first_dense > 0:                     # a dense layer exists (not in the drafter)
                @pl.when(jnp.logical_not(is_moe))
                def _dense():
                    xq, sx = quant_rows(X)
                    scr["xq"][:, :H] = xq
                    a = matmul("w13", l, scr["xq"], H, 2 * cfg.IDR)
                    gu = bf(a * sx * sm(slot, "w13_s"))
                    mid = bf(gu[:, :cfg.IDR] * jax.nn.sigmoid(gu[:, :cfg.IDR]))
                    mid = bf(mid * gu[:, cfg.IDR:])
                    xq, sx = quant_rows(mid)
                    scr["xq"][:, :cfg.IDR] = xq
                    a = matmul("w2", l, scr["xq"], cfg.IDR, H)
                    scr["part"][...] = bf(a * sx * sm(slot, "w2_s"))

            if cfg.first_dense < L:                     # a MoE layer exists (not in a dense-only build)
                @pl.when(is_moe)
                def _moe():
                    moe(l, slot, X)

            ph.mark("L_rs_mlp")
            mlp = reduce_scatter_rows(scr["part"][...], "rsm_send", "rsm_recv")
            scr["h"][...] = own_rows_pad(bf(mlp))
            if draft:
                pl.when(l == 0)(lambda: stage_wait(l))
            else:
                stage_wait(l)
            ph.end()
            if debug:
                dcs = [pltpu.make_async_copy(scr["h"].at[pl.ds(0, RPR)], dbg_h.at[l, 0],
                                             scr["dbg_sem"].at[0]),
                       pltpu.make_async_copy(scr["res"].at[pl.ds(0, RPR)], dbg_h.at[l, 1],
                                             scr["dbg_sem"].at[0]),
                       pltpu.make_async_copy(scr["own_v"], dbg_own.at[l], scr["dbg_sem"].at[1]),
                       pltpu.make_async_copy(scr["dbgl"], dbg_lse.at[l], scr["dbg_sem"].at[2]),
                       ] + ([pltpu.make_async_copy(scr["dbgo"], dbg_part.at[l], scr["dbg_sem"].at[2])]
                            if "dbgo" in scr else [])
                for d_ in dcs:
                    d_.start()
                for d_ in dcs:
                    d_.wait()
            if draft and "d_glue" not in skip:
                pd = Phases(tiles.trace)
                pd.mark("D_glue")
                draft_glue(l, slot)
                pd.end()
            return carry

        # ---------------- indexer ----------------
        def indexer(l, slot):
            IH, IDIM = cfg.IH, cfg.IDIM
            pi = Phases(tiles.trace)
            pi.mark("I_proj")
            # q: fp8 linear (replicated), rope on dims 0..DR-1, fp8 per (row, head), bf16 scale
            xq, sx = quant_rows(scr["qc"][...])
            scr["xq"][:, :cfg.QL] = xq
            a = matmul("idx_wq", l, scr["xq"], cfg.QL, IH * IDIM)
            iq = bf(a * sx * sm(slot, "idx_wq_s"))
            wts = jnp.zeros((M, 128), F32)
            # k / weights: bf16 linear (replicated) on the normed rows
            a2 = matmul("idx_wk", l, scr["xbf"], H, rup(IDIM + IH))
            kw = bf(a2)
            for h_ in range(IH):
                v = rope(iq[:, h_ * IDIM:(h_ + 1) * IDIM], rope_tab[0], rope_tab[1])
                s = bf(jnp.max(jnp.abs(v), axis=1, keepdims=True) * (1.0 / FP8_MAX))
                inv = jnp.where(s == 0, 0.0, bf(1.0 / jnp.where(s == 0, 1.0, s)))
                qv = jnp.clip(bf(v * inv), -FP8_MAX, FP8_MAX).astype(FP8).astype(F32)
                scr["iqv"][h_, 0:M, :] = qv
                # weights fold, left to right: bf16(f32(w) * q_scale * IDIM^-1/2 * IH^-1/2)
                wcol = bf(((kw[:, IDIM + h_:IDIM + h_ + 1] * s) * (IDIM ** -0.5)) * (IH ** -0.5))
                wts = jnp.where(lane128 == h_, wcol, wts)
            scr["iqv"][:, M:M + QB8, :] = jnp.zeros((IH, QB8, 128), F32)
            scr["wts"][0:M, :] = wts
            scr["wts"][M:M + QB8, :] = jnp.zeros((QB8, 128), F32)
            # k record: LayerNorm (f32, eps 1e-6) -> bf16 -> rope dims 0..DR-1 -> ue8m0 fp8
            kx = kw[:, 0:IDIM]
            mu = jnp.mean(kx, axis=1, keepdims=True)
            var = jnp.mean((kx - mu) ** 2, axis=1, keepdims=True)
            kn = bf((kx - mu) * lax.rsqrt(var + 1e-6) * sm(slot, "idx_kn_w") + sm(slot, "idx_kn_b"))
            kr = rope(kn, rope_tab[0], rope_tab[1])
            s = jnp.max(jnp.abs(kr), axis=1, keepdims=True) * (1.0 / FP8_MAX)       # f32
            sb = lax.bitcast_convert_type(s, I32)
            e = (shr(sb, 23) & 0xFF) - 127
            e = jnp.where((sb & 0x7FFFFF) != 0, e + 1, e)                    # ceil(log2 s)
            e = jnp.where(s == 0, -127, e)
            e = jnp.clip(e, -127, 127)
            sc = lax.bitcast_convert_type(shl(jnp.clip(e, -126, 127) + 127, 23), F32)
            sc = jnp.where(e == -127, 2.0 ** -127, sc)
            k8 = fp8_bytes(jnp.clip(kr / sc, -FP8_MAX, FP8_MAX).astype(FP8))       # [M, 128]
            rec = jnp.concatenate([k8, jnp.where(lax.broadcasted_iota(I32, (M, 128), 1) == 0,
                                                 e + 127, 0)], axis=1)        # [M, 256] bytes
            scr["rec_v"][...] = rec[0:TMAX]
            cp = pltpu.make_async_copy(scr["rec_v"], scr["rec_s"], scr["misc_sem"])
            cp.start()
            cp.wait()
            pltpu.make_async_copy(scr["rec_v"], irec_out.at[l], scr["kvw_sem"].at[2]).start()
            # local scan per request -> keys per row -> local top-KT -> ksend
            scr["ksend"][...] = jnp.full((TMAX, KR_, 128), INT_MIN, I32)
            pi.mark("I_scan_ltopk")

            def per_req(b, c):
                @pl.when(qmeta[b, 3] > 0)
                def _():
                    scan_request(l, b)
                return c
            lax.fori_loop(0, B, per_req, 0)
            # AG candidate keys, global top-KT, owned winners
            pi.mark("I_agk")
            scr["kall"][me] = scr["ksend"][...]
            exchange(lambda d: scr["ksend"], scr["kall"], "k_send", "k_recv", rows=T)

            pi.mark("I_gtopk")

            def per_row(t_, c):
                @pl.when(rmeta[t_, 0] > 0)
                def _():
                    global_row(t_)

                @pl.when(rmeta[t_, 0] <= 0)
                def _():
                    scr["nown_s"][t_] = jnp.int32(0)
                return c
            lax.fori_loop(0, T, per_row, 0)

            pi.mark("I_union")

            def per_req_u(b, off):
                return lax.cond(qmeta[b, 3] > 0, lambda: build_union(b, off), lambda: off)
            lax.fori_loop(0, B, per_req_u, jnp.int32(0))
            pi.mark("I_udma")
            cpu = [pltpu.make_async_copy(scr["own_u"], scr["own_uv"], scr["misc_sem"]),
                   pltpu.make_async_copy(scr["memb"], scr["memb_v"], scr["misc_sem"])]
            for c_ in cpu:
                c_.start()
            for c_ in cpu:
                c_.wait()
            pi.mark("I_recw")
            pltpu.make_async_copy(scr["rec_v"], irec_out.at[l], scr["kvw_sem"].at[2]).wait()
            pi.end()

        def scan_request(l, b):
            """Score request b's rows against this rank's local indexer keys; local top-KT per row."""
            IH, IDIM = cfg.IH, cfg.IDIM
            t0 = qmeta[b, 0]
            nq = qmeta[b, 1]
            seq = qmeta[b, 2]
            nloc = (seq // G_) * ILV + jnp.clip(seq % G_ - ILV * me, 0, ILV)
            npages = (nloc + pg - 1) >> lg_pg
            # q block: rows t0..t0+7 of every head, head-major ([IH * 8, IDIM]), and weights
            qblk = jnp.concatenate([scr["iqv"][h_, pl.ds(t0, QB8), :] for h_ in range(IH)],
                                   axis=0).astype(FP8)
            wq = scr["wts"][pl.ds(t0, QB8), :]                               # [8, 128] lanes = head
            pos_q = rvec[pl.ds(t0, QB8), 1:2]                                # [8, 1]
            q_ok = lax.broadcasted_iota(I32, (QB8, 1), 0) < nq
            SD = tiles.scan_depth
            sub_r = lax.broadcasted_iota(I32, (QB8, pg), 0)
            lane_r = lax.broadcasted_iota(I32, (QB8, pg), 1)

            def run(ref):
                def pcopy(p, sl_):
                    return pltpu.make_async_copy(ref.at[bt[b, p]], scr["ipage"].at[sl_],
                                                 scr["ip_sem"].at[sl_])
                for qq in range(SD - 1):
                    @pl.when(qq < npages)
                    def _():
                        pcopy(qq, qq).start()

                def page_body(p, c):
                    sl_ = p % SD

                    @pl.when(p + SD - 1 < npages)
                    def _():
                        pcopy(p + SD - 1, (p + SD - 1) % SD).start()
                    pcopy(0, sl_).wait()
                    # this step's new records are not in HBM yet: patch the owned ones into the page
                    lane_pg = lax.broadcasted_iota(I32, (4, pg), 1)
                    sub4 = lax.broadcasted_iota(I32, (4, pg), 0)
                    for qq in range(cfg.QMAX):
                        t2 = jnp.minimum(t0 + qq, TMAX - 1)
                        pos2 = rmeta[t2, 1]
                        lo2 = (pos2 // G_) * ILV + pos2 % ILV
                        hit = jnp.logical_and(jnp.logical_and(qq < nq, (pos2 % G_) // ILV == me),
                                              (lo2 >> lg_pg) == p)

                        @pl.when(hit)
                        def _(t2=t2, lo2=lo2):
                            sl2 = lo2 & (pg - 1)
                            for w in range(IR4):
                                v4 = [scr["rec_s"][t2, 4 * w + q_] for q_ in range(4)]
                                col = jnp.where(sub4 == 0, v4[0], jnp.where(sub4 == 1, v4[1], jnp.where(
                                    sub4 == 2, v4[2], v4[3])))
                                old = scr["ipage"][sl_, w].astype(I32)
                                scr["ipage"][sl_, w] = jnp.where(lane_pg == sl2, col, old).astype(U8)
                    blk = scr["ipage"][sl_]                                  # [IR4, 4, pg] u8
                    kT = pltpu.bitcast(blk[0:IDIM // 4].reshape(IDIM, pg), FP8)
                    scale = pltpu.bitcast(blk[IDIM // 4:IDIM // 4 + 1].reshape(4, pg)[0:1],
                                          jnp.float8_e8m0fnu).astype(BF16).astype(F32)   # [1, pg]
                    st_ = lax.dot_general(*mxu(qblk, kT, tiles), NN,
                                          preferred_element_type=F32)        # [IH*8, pg]
                    acc = jnp.maximum(st_[0:QB8], 0.0) * wq[:, 0:1]
                    for h_ in range(1, IH):
                        acc = acc + jnp.maximum(st_[h_ * QB8:(h_ + 1) * QB8], 0.0) * wq[:, h_:h_ + 1]
                    score = acc * scale
                    lidx = p * pg + lane_r
                    g = (lidx // ILV) * G_ + me * ILV + lidx % ILV
                    ok = (g < seq) & (g <= pos_q) & q_ok & (score != -jnp.inf)
                    key = jnp.where(ok, mono_key(score), jnp.int32(INT_MIN))     # [8, pg]
                    for qq in range(cfg.QMAX):
                        rows_ = jnp.concatenate([key[qq:qq + 1, j * 128:(j + 1) * 128]
                                                 for j in range(NPR)], axis=0)  # [NPR, 128]
                        scr["keys"][qq, pl.ds(p * NPR, NPR), :] = rows_
                    return c
                lax.fori_loop(0, npages, page_body, 0)
            csel(("idx",), l, run)

            # local top-KT per row of the request
            for qq in range(cfg.QMAX):
                @pl.when(qq < nq)
                def _(qq=qq):
                    t_ = t0 + qq
                    pos = rmeta[t_, 1]
                    L_ = pos + 1
                    nval = (L_ // G_) * ILV + jnp.clip(L_ % G_ - ILV * me, 0, ILV)
                    topk2.local_topk(scr["keys"].at[qq], nval, KT, scr["cidx"], scr["ksend"].at[t_],
                                     scr["kpre"], scr["tk_acc"], scr["tk_cc"], kc=tiles.topk_kc)
                    scr["cand_v"][t_] = scr["cidx"][...]
                    cp = pltpu.make_async_copy(scr["cidx"], scr["cand_s"].at[pl.ds(t_ * KR_, KR_)],
                                               scr["misc_sem"])
                    cp.start()
                    cp.wait()

        def build_union(b, off, rn=None):
            """Request b's rows' owned winner lists (ascending local indices) -> their merged union at
            flat rows [off, off + ceil(nu/128)) of own_u, with memb = the bitmask of the request's rows
            (bit qq) holding each entry; nu_s[b], uo_s[b] = (nu, off). Returns the next free row.
            Sum over requests of ceil(nu/128) <= T*KR_ + B: the flat arrays never overflow."""
            t0 = qmeta[b, 0]
            nq = qmeta[b, 1]
            QM = cfg.QMAX
            BIG = jnp.int32(2 ** 30)
            t0r, _, p0r, noldr = req_new(b) if rn is None else rn
            ns = [jnp.where(qq < nq, scr["nown_s"][jnp.minimum(t0 + qq, TMAX - 1)], 0) for qq in range(QM)]
            tot = ns[0]
            for x in ns[1:]:
                tot = tot + x

            def head(qq, p):
                t_ = jnp.minimum(t0 + qq, TMAX - 1)
                v = scr["own_s"][t_ * KR_ + (jnp.minimum(p, KT - 1) >> 7), jnp.minimum(p, KT - 1) & 127]
                return jnp.where(p < ns[qq], v, BIG)

            def cond(c):
                return c[0] < tot                    # consumed count < total entries

            def body(c):
                consumed, k = c[0], c[1]
                ps = list(c[3:])
                hs = [head(qq, ps[qq]) for qq in range(QM)]
                m = hs[0]
                for h in hs[1:]:
                    m = jnp.minimum(m, h)
                bits = jnp.int32(0)
                new_ps = []
                for qq in range(QM):
                    hit = (hs[qq] == m).astype(I32)
                    bits = bits | (hit << qq)
                    new_ps.append(ps[qq] + hit)
                # the entry's gather source, resolved once per fresh top-k (reuse layers share it):
                # in HBM -> page << 8 | slot; this step's own new row (sorted last) -> staging row << 2
                # | seg. Both keep the rope segment (local index & 3) in the low bits.
                old_ = m < noldr
                g = (m // ILV) * G_ + me * ILV + m % ILV
                t2 = jnp.clip(t0r + (g - p0r), 0, TMAX - 1)
                src = jnp.where(old_, (bt[b, jnp.where(old_, m, 0) >> lg_pg] << 8) | (m & (pg - 1)),
                                (t2 << 2) | (m & 3))
                p = off * 128 + k
                scr["own_u"][p >> 7, p & 127] = src
                scr["memb"][p >> 7, p & 127] = bits
                took = bits & 1
                for qq in range(1, QM):
                    took = took + ((bits >> qq) & 1)
                return (consumed + took, k + 1, c[2] + old_.astype(I32), *new_ps)
            res = lax.while_loop(cond, body, (jnp.int32(0),) * 3 + (jnp.int32(0),) * QM)
            scr["nu_s"][b] = res[1]
            scr["uo_s"][b] = off
            scr["nuo_s"][b] = res[2]
            return off + (res[1] + 127) // 128

        def global_row(t_):
            """Row t_'s global top-KT over all ranks' candidates -> own_s / own_v / nown_s."""
            for s_ in range(R):
                scr["gk"][s_ * KR_:(s_ + 1) * KR_, :] = scr["kall"][s_, t_]
            if GKR > R * KR_:
                scr["gk"][R * KR_:GKR, :] = jnp.full((GKR - R * KR_, 128), INT_MIN, I32)
            # owned winners' local indices compacted by the vector extraction itself (payload = the
            # candidate's local index): no per-entry scalar gather (a rank owning most winners was a
            # multi-ms straggler, PHASES.draft1.md)
            n_own = topk2.global_owned(scr["gk"], me, KT, scr["oslot"], scr["kpre"],
                                       scr["tk_acc"].at[pl.ds(0, min(tiles.topk_kc_global, KT))],
                                       scr["tk_cc"], scr["gmk"], kc=tiles.topk_kc_global,
                                       pay_ref=scr["cand_v"].at[t_], pay_out=scr["own_v"].at[t_])
            kio = (lax.broadcasted_iota(I32, (KR_, 128), 0) * 128 + lax.broadcasted_iota(I32, (KR_, 128), 1))
            scr["own_v"][t_] = jnp.where(kio < n_own, scr["own_v"][t_], -1)
            scr["nown_s"][t_] = n_own
            cp = pltpu.make_async_copy(scr["own_v"].at[t_], scr["own_s"].at[pl.ds(t_ * KR_, KR_)],
                                       scr["misc_sem"])
            cp.start()
            cp.wait()

        # ---------------- attention ----------------
        QN = cfg.QMAX * NH                                   # a request's rows x heads (MXU M dim)

        def nu_of(b):
            b_c = jnp.minimum(b, BMAX - 1)
            return jnp.where(jnp.logical_and(b < B, qmeta[b_c, 3] > 0), scr["nu_s"][b_c], 0)

        def tile_n(b, j):
            return jnp.clip(nu_of(b) - j * KT, 0, KT)

        def attn_issue(l, b, buf, j):
            """Start the gather of tile j (union entries j*KT ..) of request b into buffer buf: HBM
            rows (page, slot resolved by build_union), then this step's own new rows from the VMEM
            staging (not in HBM yet; they sort last)."""
            b_c = jnp.minimum(b, BMAX - 1)
            n = tile_n(b, j)
            base = scr["uo_s"][b_c] * 128 + j * KT
            n_old = jnp.clip(jnp.where(b < B, scr["nuo_s"][b_c], 0) - j * KT, 0, n)

            def go(rn, rr):
                def obody(k, c2):
                    p = base + k
                    v = scr["own_u"][p >> 7, p & 127]
                    page, sl = v >> 8, v & 255
                    pltpu.make_async_copy(rn.at[page, pl.ds(sl, 1)], scr["g_nope"].at[buf, pl.ds(k, 1)],
                                          scr["g_sem"].at[buf, 0]).start()
                    pltpu.make_async_copy(rr.at[page, pl.ds(sl >> 2, 1)],
                                          scr["g_rope"].at[buf, pl.ds(k, 1)], scr["g_sem"].at[buf, 1]).start()
                    return c2
                lax.fori_loop(0, n_old, obody, 0)

                def nbody(k, c2):
                    p = base + k
                    t2 = scr["own_u"][p >> 7, p & 127] >> 2
                    pltpu.make_async_copy(scr["wrow"].at[pl.ds(t2, 1)], scr["g_nope"].at[buf, pl.ds(k, 1)],
                                          scr["g_sem"].at[buf, 0]).start()
                    pltpu.make_async_copy(scr["rwrow"].at[pl.ds(t2, 1)], scr["g_rope"].at[buf, pl.ds(k, 1)],
                                          scr["g_sem"].at[buf, 1]).start()
                    return c2
                lax.fori_loop(n_old, n, nbody, 0)
            if "gather" not in skip:
                csel(("nope", "rope"), l, go)

        def attn_first(l):
            """Request 0's first tile: issued as early as its inputs allow (after the indexer on
            fresh top-k layers, after the staging otherwise) so it flies under the q all-gather."""
            @pl.when(B > 0)
            def _():
                attn_issue(l, jnp.int32(0), 0, jnp.int32(0))

        def attention(l):
            """Per request: its rows' q stacked on the MXU M axis ([QMAX*NH, .]); over tiles of the
            union of its rows' owned winners (gathered + nope-decoded once per request), online
            softmax with each row restricted to its own winners (membership bits) -> per row
            osend[d, t]: the bf16 normalized partial of heads d*HR.. packed in u32 pairs | lse (f32
            bits) in the tail lanes (lse = m + ln l; -inf if this rank owns none of the row's winners)."""
            lane_h = lax.broadcasted_iota(I32, (NH, 128), 1)
            row_q = lax.broadcasted_iota(I32, (QN, 1), 0) // NH
            NEG = jnp.finfo(F32).min

            def wait(b, buf, j):
                n = tile_n(b, j)
                ref0 = C["nope"][next(iter(C["nope"]))]
                rref0 = C["rope"][next(iter(C["rope"]))]

                def wbody(k, c2):                        # one wait per issued row (same bytes each)
                    pltpu.make_async_copy(ref0.at[0, pl.ds(0, 1)], scr["g_nope"].at[buf, pl.ds(0, 1)],
                                          scr["g_sem"].at[buf, 0]).wait()
                    pltpu.make_async_copy(rref0.at[0, pl.ds(0, 1)], scr["g_rope"].at[buf, pl.ds(0, 1)],
                                          scr["g_sem"].at[buf, 1]).wait()
                    return c2
                if "gather" not in skip:
                    lax.fori_loop(0, n, wbody, 0)

            def qload(t0):
                """q of the request's QMAX rows (rows past nq: whatever is there; masked later)."""
                for qq in range(cfg.QMAX):
                    t_ = jnp.minimum(t0 + qq, TMAX - 1)
                    heads = []
                    for s_ in range(R):
                        u = scr["qall"][s_, pl.ds(t_, 1), :]                 # [1, QH] u32
                        full = jnp.concatenate([unpack_lo(u), unpack_hi(u)], axis=1)
                        heads += [full[:, hh * QA:(hh + 1) * QA] for hh in range(HR)]
                    Q = jnp.concatenate(heads, axis=0)                       # [NH, QA] f32
                    r0 = qq * NH
                    scr["qb"][r0:r0 + NH, :] = Q[:, 0:KL].astype(BF16)
                    xr = Q[:, KL:QA]                                         # rope dims in lanes 0..DR-1
                    q0 = jnp.where(lane_h < 32, xr, 0.0)                      # dims 0..31
                    q1 = jnp.where(lane_h < 32, pltpu.roll(xr, 96, 1), 0.0)   # dims 32..63
                    for sg in range(4):
                        scr["q0b"][sg, r0:r0 + NH, :] = (pltpu.roll(q0, 32 * sg, 1) if sg else q0).astype(BF16)
                        scr["q1b"][sg, r0:r0 + NH, :] = (pltpu.roll(q1, 32 * sg, 1) if sg else q1).astype(BF16)

            def tile(b, buf, j, nu, pa):
                gn, gr = scr["g_nope"].at[buf], scr["g_rope"].at[buf]
                base = j * KT
                nc = jnp.where("attn" in skip, 0, (tile_n(b, j) + 127) // 128)
                urow = scr["uo_s"][jnp.minimum(b, BMAX - 1)] + base // 128
                pa.mark("A_decode")

                def d_body(ci, c2):                      # nope bands, once per request
                    r0 = pl.multiple_of(ci * 128, 128)
                    w = lax.bitcast_convert_type(gn[pl.ds(r0, 128), :], I32)
                    for b_ in range(4):
                        scr["kvb"][pl.ds(r0, 128), b_ * 128:(b_ + 1) * 128] = fp8_decode_hw(shr(w, 8 * b_) & 0xFF)
                    return c2
                lax.fori_loop(0, nc, d_body, 0)
                pa.mark("A_score")

                def c_body(ci, c2):
                    r0 = pl.multiple_of(ci * 128, 128)
                    s = jnp.zeros((QN, 128), F32)
                    for b_ in range(4):
                        s = s + lax.dot_general(scr["qb"][:, b_ * 128:(b_ + 1) * 128],
                                                scr["kvb"][pl.ds(r0, 128), b_ * 128:(b_ + 1) * 128], NT,
                                                preferred_element_type=F32)
                    rw_ = lax.bitcast_convert_type(gr[pl.ds(r0, 128), :], I32)
                    B0 = fp8_decode_hw(rw_ & 0xFF)
                    B1 = fp8_decode_hw(shr(rw_, 8) & 0xFF)
                    iv = scr["own_uv"][pl.ds(urow + ci, 1), :]               # [1, 128] local idx
                    seg = iv & 3
                    for sg in range(4):
                        ss = (lax.dot_general(scr["q0b"][sg], B0, NT, preferred_element_type=F32)
                              + lax.dot_general(scr["q1b"][sg], B1, NT, preferred_element_type=F32))
                        s = s + jnp.where(seg == sg, ss, 0.0)
                    s = s * sm_scale
                    bits = scr["memb_v"][pl.ds(urow + ci, 1), :]             # [1, 128] row bits
                    kk = base + ci * 128 + lax.broadcasted_iota(I32, (1, 128), 1)
                    mask = jnp.logical_and(((bits >> row_q) & 1) == 1, kk < nu)   # [QN, 128]
                    s = jnp.where(mask, s, NEG)
                    m_old = scr["m_s"][...]
                    m_new = jnp.maximum(m_old, jnp.max(s, axis=1, keepdims=True))
                    alpha = jnp.exp(m_old - m_new)
                    p = jnp.where(mask, jnp.exp(s - m_new), 0.0)
                    kv = scr["kvb"][pl.ds(r0, 128), :].astype(F32)
                    scr["l_s"][...] = alpha * scr["l_s"][...] + jnp.sum(p, axis=1, keepdims=True)
                    scr["att_acc"][...] = alpha * scr["att_acc"][...] + lax.dot_general(p, kv, NN,
                                                                                preferred_element_type=F32)
                    scr["m_s"][...] = m_new
                    return c2
                lax.fori_loop(0, nc, c_body, 0)

            def emit(t_, qq):
                n = jnp.where(rmeta[t_, 0] > 0, scr["nown_s"][t_], 0)
                r0 = qq * NH
                lsum = scr["l_s"][r0:r0 + NH, :]
                mx = scr["m_s"][r0:r0 + NH, :]
                pv = scr["att_acc"][r0:r0 + NH, :]
                has = n > 0
                out = jnp.where(has, bf(pv * pl.reciprocal(jnp.maximum(lsum, 1e-30), approx=True)), 0.0)
                lse = jnp.where(has, mx + jnp.log(jnp.maximum(lsum, 1e-30)), -jnp.inf)    # [NH, 1]
                lse_b = jnp.broadcast_to(lse, (NH, 128))
                if NH < 128:
                    lse_b = jnp.concatenate([lse_b, jnp.full((128 - NH, 128), -jnp.inf, F32)], axis=0)
                lse_row = jnp.transpose(lse_b)[0:1]                          # [1, 128], lane = head
                if debug:
                    scr["dbgl"][pl.ds(t_, 1), :] = lse_row
                    if "dbgo" in scr:
                        scr["dbgo"][t_] = out
                for d in range(R):
                    od = jnp.concatenate([out[d * HR + hh:d * HR + hh + 1] for hh in range(HR)], axis=1)
                    lrow = pltpu.roll(lse_row, (128 - d * HR) % 128, 1) if d else lse_row
                    scr["osend"][d, pl.ds(t_, 1), :] = jnp.concatenate(
                        [pack2(od[:, :OH], od[:, OH:]), lax.bitcast_convert_type(lrow, U32)], axis=1)

            def request(b, c):
                pa = Phases(tiles.trace)
                buf = b % 2
                nu = nu_of(b)
                b_c = jnp.minimum(b, BMAX - 1)
                t0 = qmeta[b_c, 0]
                nq = qmeta[b_c, 1]
                pa.mark("A_qasm")
                qload(t0)                                # under the in-flight gather
                scr["m_s"][...] = jnp.full((QN, 1), NEG, F32)
                scr["l_s"][...] = jnp.zeros((QN, 1), F32)
                scr["att_acc"][...] = jnp.zeros((QN, KL), F32)
                pa.mark("A_gather")
                wait(b, buf, 0)

                @pl.when(b + 1 < B)
                def _():
                    attn_issue(l, b + 1, 1 - buf, jnp.int32(0))   # flies under this request's compute
                tile(b, buf, jnp.int32(0), nu, pa)

                def more(j, c2):                         # union > KT (top-k skewed onto this rank)
                    pa.mark("A_gather")
                    attn_issue(l, b, buf, j)
                    wait(b, buf, j)
                    tile(b, buf, j, nu, pa)
                    return c2
                lax.fori_loop(1, (nu + KT - 1) // KT, more, 0)
                pa.mark("A_send")
                for qq in range(cfg.QMAX):
                    @pl.when(qq < nq)
                    def _(qq=qq):
                        emit(t0 + qq, qq)
                pa.end()
                return c

            lax.fori_loop(0, B, request, 0)

        def merge():
            """LSE merge of the R partials of this rank's HR heads, main's formula
            (cp_mla_attention.py:148-186): shift = max finite lse; w_r = exp(l_r - shift) / sum;
            o = sum f32(o_r) * bf16(w_r) + sum f32(o_r) * (w_r - bf16(w_r)), then bf16."""
            os_, lr = [], []
            for s_ in range(R):
                blk = scr["orecv"][s_]                                       # [TMAX, OW] u32
                os_.append(jnp.concatenate([unpack_lo(blk[:, :OH]), unpack_hi(blk[:, :OH])], axis=1))
                lr.append(lax.bitcast_convert_type(blk[:, OH:OH + 128], F32))  # lanes hh
            fin = [jnp.isfinite(x_) for x_ in lr]
            shift = jnp.full((TMAX, 128), -jnp.inf, F32)
            for x_, f_ in zip(lr, fin):
                shift = jnp.maximum(shift, jnp.where(f_, x_, -jnp.inf))
            shift = jnp.where(jnp.isfinite(shift), shift, 0.0)
            ws = [jnp.where(f_, jnp.exp(x_ - shift), 0.0) for x_, f_ in zip(lr, fin)]
            tot = ws[0]
            for w_ in ws[1:]:
                tot = tot + w_
            tot = jnp.where(tot > 0, tot, 1.0)
            ws = [w_ / tot for w_ in ws]
            valid_t = rvalid[0:TMAX]
            outs = []
            for hh in range(HR):
                hi = jnp.zeros((TMAX, KL), F32)
                lo = jnp.zeros((TMAX, KL), F32)
                for s_ in range(R):
                    o_ = os_[s_][:, hh * KL:(hh + 1) * KL]                   # [TMAX, KL]
                    w_ = ws[s_][:, hh:hh + 1]
                    wh = bf(w_)
                    hi = hi + o_ * wh
                    lo = lo + o_ * (w_ - wh)
                o = bf(hi + lo) * valid_t
                if M > TMAX:
                    o = jnp.concatenate([o, jnp.zeros((M - TMAX, KL), F32)], axis=0)
                outs.append(o)
            return outs

        # ---------------- MoE ----------------
        def moe(l, slot, X):
            ER = cfg.ER
            pm = Phases(tiles.trace)
            pm.mark("M_sw13")
            KREP, NS = cfg.KREP, cfg.R * cfg.KREP

            def es_dst(k, rep_):
                if not KREP:
                    return scr[k]
                return scr[k].at[pl.ds(ER, KREP)] if rep_ else scr[k].at[pl.ds(0, ER)]
            wsel("es13", l, lambda r: pltpu.make_async_copy(r, es_dst("es13", False), scr["es_sem"].at[0]).start())
            wsel("es2", l, lambda r: pltpu.make_async_copy(r, es_dst("es2", False), scr["es_sem"].at[1]).start())
            if KREP:
                wsel("rs13", l, lambda r: pltpu.make_async_copy(r, es_dst("es13", True), scr["es_sem"].at[0]).start())
                wsel("rs2", l, lambda r: pltpu.make_async_copy(r, es_dst("es2", True), scr["es_sem"].at[1]).start())
            xq, sx = quant_rows(X)
            scr["xq"][:, :H] = xq
            # shared expert gate_up (TP slice)
            a = matmul("sw13", l, scr["xq"], H, 2 * cfg.ISR)
            gu = bf(a * sx * sm(slot, "sw13_s"))
            mid = bf(gu[:, :cfg.ISR] * jax.nn.sigmoid(gu[:, :cfg.ISR]))
            mid = bf(mid * gu[:, cfg.ISR:])
            pm.mark("M_route")
            # router: bf16 logits -> f32 sigmoid + bias -> top-k (ties lowest id) -> renorm -> bf16
            logits = bf(matmul("gate", l, scr["xbf"], H, cfg.E))
            scv = jax.nn.sigmoid(logits)
            choice = scv + sm(slot, "gate_bias")
            eio = lax.broadcasted_iota(I32, (M, cfg.E), 1)
            wsel_ = jnp.zeros((M, cfg.E), F32)
            for _ in range(cfg.TOPK):
                mxv = jnp.max(choice, axis=1, keepdims=True)
                am = jnp.min(jnp.where(choice == mxv, eio, cfg.E), axis=1, keepdims=True)
                sel = eio == am
                wsel_ = jnp.where(sel, scv, wsel_)
                choice = jnp.where(sel, -jnp.inf, choice)
            wsum = jnp.sum(wsel_, axis=1, keepdims=True)
            wn = bf(wsel_ / jnp.maximum(wsum, 1e-20)) * rvalid
            wloc = jnp.zeros((M, 128), F32)
            hitv = jnp.zeros((1, 128), I32)
            for j in range(ER):
                e = me * ER + j
                col = jnp.sum(jnp.where(eio == e, wn, 0.0), axis=1, keepdims=True)
                cnt = jnp.sum(jnp.where(eio == e, (wsel_ > 0).astype(I32), 0) * (rvalid > 0).astype(I32))
                wloc = jnp.where(lane128 == j, col, wloc)
                hitv = jnp.where(lane128 == j, cnt, hitv)
            valid_tok = rvalid > 0
            if KREP:
                # replica table -> hits rows 2..7: 2-3 expert hit bits (all E), 4 rep_e, 5 rep_s,
                # 6 per-rank distinct-expert loads, 7 moved flags. Routing is replicated, so every
                # rank computes the same assignment (DP16 kernel, replicas.py).
                hitf = jnp.sum(jnp.where(jnp.logical_and(wsel_ > 0, valid_tok), 1.0, 0.0),
                               axis=0, keepdims=True)                      # [1, E]
                hitb = (hitf > 0).astype(I32)
                erow = lax.broadcasted_iota(I32, (1, cfg.E), 1)
                for c_ in range(2):
                    lo_, hi_ = c_ * 128, min(cfg.E, (c_ + 1) * 128)
                    if hi_ <= lo_:
                        row_ = jnp.zeros((1, 128), I32)
                    elif hi_ - lo_ == 128:
                        row_ = hitb[:, lo_:hi_]
                    else:
                        row_ = jnp.zeros((1, 128), I32)
                        for e_ in range(lo_, hi_):
                            row_ = jnp.where(lane128 == e_ - lo_,
                                             jnp.sum(jnp.where(erow == e_, hitb, 0)), row_)
                    scr["hitv"][2 + c_:3 + c_, :] = row_
                rep_e = sm(slot, "rep_e").astype(I32)
                rep_s = sm(slot, "rep_s").astype(I32)
                scr["hitv"][4:5, :] = rep_e
                scr["hitv"][5:6, :] = rep_s
                ld_ = jnp.zeros((1, 128), I32)
                for r_ in range(R):
                    ld_ = jnp.where(lane128 == r_, jnp.sum(jnp.where(erow // ER == r_, hitb, 0)), ld_)
                scr["hitv"][6:7, :] = ld_
                scr["hitv"][7:8, :] = jnp.zeros((1, 128), I32)
                ent = lane128 < NS
                for q_ in range(KREP):                   # wloc lane ER + q: own slot q's expert
                    eq = jnp.sum(jnp.where(jnp.logical_and(ent, rep_s == me * KREP + q_), rep_e, 0),
                                 axis=1, keepdims=True)
                    col = jnp.sum(jnp.where(eio == eq, wn, 0.0), axis=1, keepdims=True)
                    wloc = jnp.where(lane128 == ER + q_, col, wloc)
            scr["wloc"][...] = wloc
            scr["hitv"][0:1, :] = hitv
            cp = pltpu.make_async_copy(scr["hitv"], scr["hits"], scr["misc_sem"])
            cp.start()
            cp.wait()

            if not KREP:
                def cbody(j, n):
                    h_ = scr["hits"][0, j] > 0

                    @pl.when(h_)
                    def _():
                        scr["hits"][1, n] = j
                    return n + h_.astype(I32)
                n_hit = lax.fori_loop(0, ER, cbody, jnp.int32(0))
            else:
                hs = scr["hits"]

                def hrow(e):
                    return 2 + e // 128, e % 128

                def gbody(t, c):
                    # greedy, 3 passes over the table: move hit expert e (not yet moved) to its
                    # replica holder iff that leaves holder < owner in distinct-expert visits
                    s_ = t % NS
                    e = hs[4, s_]
                    sl = hs[5, s_]
                    ok = jnp.logical_and(e >= 0, sl >= 0)
                    e = jnp.maximum(e, 0)
                    r_ = jnp.maximum(sl, 0) // KREP
                    a_ = e // ER
                    la = hs[6, a_]
                    lr = hs[6, r_]
                    er_, el_ = hrow(e)
                    mv = jnp.logical_and(ok, jnp.logical_and(hs[er_, el_] == 1, lr + 1 < la))

                    @pl.when(mv)
                    def _():
                        hs[6, a_] = la - 1
                        hs[6, r_] = lr + 1
                        hs[er_, el_] = jnp.int32(2)
                        hs[7, s_] = jnp.int32(1)
                    return c
                lax.fori_loop(0, 3 * NS, gbody, 0)

                def pbody(j, n):                         # own primaries still here (hit, not moved)
                    er_, el_ = hrow(me * ER + j)
                    h_ = hs[er_, el_] == 1

                    @pl.when(h_)
                    def _():
                        hs[1, n] = j
                    return n + h_.astype(I32)
                n_hit = lax.fori_loop(0, ER, pbody, jnp.int32(0))

                def rbody(s_, n):                        # experts moved onto own replica slots
                    sl = hs[5, s_]
                    h_ = jnp.logical_and(hs[7, s_] == 1, jnp.logical_and(sl >= 0, sl // KREP == me))

                    @pl.when(h_)
                    def _():
                        hs[1, n] = ER + sl % KREP
                    return n + h_.astype(I32)
                n_hit = lax.fori_loop(0, NS, rbody, n_hit)
            if "experts" in skip:
                n_hit = jnp.int32(0)
            st[6] = n_hit
            st[5] = l
            pump("gate")
            pm.mark("M_sw2")
            # shared down (local K, per-rank scale), the routed tiles stream under it
            xq2, sx2 = quant_rows(mid)
            scr["xq"][:, :cfg.ISR] = xq2
            a = matmul("sw2", l, scr["xq"], cfg.ISR, H)
            scr["shared"][...] = bf(a * sx2 * sm(slot, "sw2_s"))
            pm.mark("M_exp")
            # routed experts
            pltpu.make_async_copy(es_dst("es13", False), es_dst("es13", False), scr["es_sem"].at[0]).wait()
            pltpu.make_async_copy(es_dst("es2", False), es_dst("es2", False), scr["es_sem"].at[1]).wait()
            if KREP:
                pltpu.make_async_copy(es_dst("es13", True), es_dst("es13", True), scr["es_sem"].at[0]).wait()
                pltpu.make_async_copy(es_dst("es2", True), es_dst("es2", True), scr["es_sem"].at[1]).wait()
            scr["xqm"][...] = blk_quant(X, NBLK13, "sxb")
            scr["routed"][...] = jnp.zeros((M, H), F32)

            def ebody(i, c):
                j = scr["hits"][1, i]
                a = matmul_blk("ew13", l, scr["xqm"], H, 2 * cfg.I, j, "sxb")
                gu = bf(a * scr["es13"][pl.ds(j, 1), :])
                mid_ = bf(gu[:, :cfg.I] * jax.nn.sigmoid(gu[:, :cfg.I]))
                mid_ = bf(mid_ * gu[:, cfg.I:])
                scr["xq"][:, :cfg.I] = blk_quant(mid_, NBLK2, "sxb2")
                a = matmul_blk("ew2", l, scr["xq"], cfg.I, H, j, "sxb2")
                y_ = bf(a * scr["es2"][pl.ds(j, 1), :])
                wcol = jnp.sum(jnp.where(lane128 == j, scr["wloc"][...], 0.0), axis=1, keepdims=True)
                scr["routed"][...] += y_ * wcol
                return c
            lax.fori_loop(0, n_hit, ebody, 0)
            routed = bf(bf(scr["routed"][...]) * cfg.routed_scale)
            scr["part"][...] = bf(scr["shared"][...] + routed)
            pm.end()

        # ---------------- MTP drafter glue (draft=True; WHITEBOARD.v4) ----------------
        def index_share():
            """Pass 1's rows (one per request) take pass 0's owned lists of row P_i (IndexShare), as
            vector row copies; each request's union is its one row, resolved to gather sources with
            vector ops in an aligned KR_-row region (old entries first: the lists ascend in local
            index), then one VMEM -> SMEM copy for the issue loop. In place, ascending: P_j >= j, so
            a source row is never overwritten before it is read. No per-entry scalar loop (the
            scalar copy + build_union made the rank owning most winners a 10 ms straggler)."""
            kio = (lax.broadcasted_iota(I32, (KR_, 128), 0) * 128 + lax.broadcasted_iota(I32, (KR_, 128), 1))

            def cp(i, c):
                src = dmeta[0, i]
                n = scr["nown_s"][src]
                scr["own_v"][i] = scr["own_v"][src]
                scr["nown_s"][i] = n
                t0r, _, p0r, noldr = req_new(i, lambda r_, k_: qmeta_a[r_, k_])
                lo = scr["own_v"][i]                                          # [KR_, 128] local idx
                valid = kio < n
                old_ = jnp.logical_and(valid, lo < noldr)
                pidx = jnp.where(old_, lo >> lg_pg, 0)
                npg = (noldr + pg - 1) // pg + 1                              # live local pages

                def pg_body(j, pv):
                    return jnp.where(pidx == j, bt[i, j], pv)
                page = lax.fori_loop(0, jnp.minimum(npg, max_pages), pg_body, jnp.zeros((KR_, 128), I32))
                g = (lo // ILV) * G_ + me * ILV + lo % ILV
                t2 = jnp.clip(t0r + (g - p0r), 0, TMAX - 1)
                srcv = jnp.where(old_, shl(page, 8) | (lo & (pg - 1)), shl(t2, 2) | (lo & 3))
                r0 = pl.multiple_of(i * KR_, 8)
                scr["own_uv"][pl.ds(r0, KR_), :] = jnp.where(valid, srcv, 0)
                scr["memb_v"][pl.ds(r0, KR_), :] = jnp.where(valid, 1, 0)
                scr["nu_s"][i] = n
                scr["uo_s"][i] = i * KR_
                scr["nuo_s"][i] = jnp.sum(old_.astype(I32))
                return c
            lax.fori_loop(0, B, cp, 0)

            def zr(t_, c):
                scr["nown_s"][t_] = jnp.int32(0)
                return c
            lax.fori_loop(B, TMAX, zr, 0)
            c_ = pltpu.make_async_copy(scr["own_uv"], scr["own_u"], scr["misc_sem"])
            c_.start()
            c_.wait()

        def hbm_tile(src):
            c_ = pltpu.make_async_copy(src, scr["lmb"], scr["lm_sem"])
            c_.start()
            c_.wait()
            return scr["lmb"][...]

        def draft_prologue(tok_of, valid_of, prev, slot):
            """x = eh_proj([enorm(embed(tok)); hnorm(prev)]) -> this rank's owner rows of h (res = 0).
            embed: the owning vocab shard DMAs the row, RS sums the (one-hot) partials onto owners,
            AG to all ranks; eh_proj: this rank's H/R output columns, RS of the column blocks."""
            scr["demb"][...] = jnp.zeros((1, M, H // 2), U32)
            for t_ in (range(TMAX) if "d_emb" not in skip else ()):
                tok = tok_of(t_)

                @pl.when(jnp.logical_and(valid_of(t_), tok // VR == me))
                def _(t_=t_, tok=tok):
                    c_ = pltpu.make_async_copy(emb_h.at[pl.ds(tok - me * VR, 1)], scr["demb"].at[0, pl.ds(t_, 1)],
                                               scr["lm_sem"])
                    c_.start()
                    c_.wait()
            w_ = scr["demb"][0]                                                  # [M, H/2] u32 pairs
            e_own = reduce_scatter_rows(jnp.concatenate([unpack_lo(w_), unpack_hi(w_)], axis=1),
                                        "rso_send", "rso_recv")
            E = gather_rows_all(own_rows_pad(e_own), "ag_send", "ag_recv")          # [M, H]
            en = rmsnorm(E, sm(slot, "enorm"), cfg.eps).astype(BF16)
            hn_ = rmsnorm(prev, sm(slot, "hnorm"), cfg.eps).astype(BF16)
            ys = []
            for c in (range(DH // 128) if "d_eh" not in skip else ()):
                w0 = hbm_tile(eht_h.at[0, pl.ds(c * 128, 128)])
                a = lax.dot_general(en, w0, NT, preferred_element_type=F32)
                w1 = hbm_tile(eht_h.at[1, pl.ds(c * 128, 128)])
                a = a + lax.dot_general(hn_, w1, NT, preferred_element_type=F32)
                ys.append(bf(a))
            y = jnp.concatenate(ys, axis=1) if ys else jnp.zeros((M, DH), F32)  # [M, DH]
            col = lax.broadcasted_iota(I32, (M, H), 1) // DH
            yfull = jnp.where(col == me, jnp.concatenate([y] * R, axis=1), 0.0)
            h_own = reduce_scatter_rows(yfull, "rsm_send", "rsm_recv")
            scr["h"][...] = own_rows_pad(h_own)
            scr["res"][...] = jnp.zeros((RO8, H), F32)

        def draft_glue(l, slot):
            """After pass l: h = res + block (bf16), shared-head norm, the rows whose logits count,
            lm_head over this rank's vocab shard -> first max -> AG -> global first max (rank-major
            vocab); then pass l+1's prologue."""
            x = scr["h"][...] + scr["res"][...]
            hn = rmsnorm(bf(x), sm(slot, "shead"), cfg.eps) * own_valid()
            X = gather_rows_all(hn, "ag_send", "ag_recv").astype(BF16)          # [M, H]
            i0 = lax.broadcasted_iota(I32, (XR, 1), 0)
            selc = jnp.zeros((XR, 1), I32)
            for b_ in range(XR):
                selc = jnp.where(i0 == b_, dmeta[l, b_], selc)
            oh = jnp.where(jnp.logical_and(lax.broadcasted_iota(I32, (XR, M), 1) == selc, i0 < B), 1.0, 0.0)
            xs = lax.dot_general(oh.astype(BF16), X, NN, preferred_element_type=F32)   # exact
            scr["carry"][...] = xs
            xsb = xs.astype(BF16)
            lane_t = lax.broadcasted_iota(I32, (1, 128), 1)

            def lm_body(j, c):
                bv, bi = c
                lg = bf(lax.dot_general(xsb, hbm_tile(lm_h.at[pl.ds(j * 128, 128)]), NT,
                                        preferred_element_type=F32))             # [XR, 128]
                gi = (j * 128 + lane_t).astype(F32)      # exact (< 2^24); f32 lane reductions
                lg = jnp.where(gi < VR, lg, -jnp.inf)
                tv = jnp.max(lg, axis=1, keepdims=True)
                ti = jnp.min(jnp.where(lg == tv, gi, 3e7), axis=1, keepdims=True)
                take_ = tv > bv
                return jnp.where(take_, tv, bv), jnp.where(take_, ti, bi)
            bv, bi = lax.fori_loop(0, 0 if "d_lm" in skip else VRP // 128, lm_body,
                                   (jnp.full((XR, 1), -jnp.inf, F32), jnp.zeros((XR, 1), F32)))
            lane_x = lax.broadcasted_iota(I32, (XR, 128), 1)
            pay = jnp.where(lane_x == 0, jnp.broadcast_to(bv, (XR, 128)),
                            jnp.broadcast_to(bi + jnp.float32(VR) * me.astype(F32), (XR, 128)))
            scr["dsend"][...] = lax.bitcast_convert_type(pay, I32)   # lane 0 value, lane 1 index (f32)
            scr["drecv"][me] = scr["dsend"][...]
            exchange(lambda d: scr["dsend"], scr["drecv"], "d_send", "d_recv")
            gv, gf = jnp.full((XR, 1), -jnp.inf, F32), jnp.zeros((XR, 1), F32)
            for s_ in range(R):                          # rank-major vocab: strict > keeps the first
                blk = lax.bitcast_convert_type(scr["drecv"][s_], F32)          # [XR, 128]
                v_ = jnp.sum(jnp.where(lane_x == 0, blk, 0.0), axis=1, keepdims=True)
                i_ = jnp.sum(jnp.where(lane_x == 1, blk, 0.0), axis=1, keepdims=True)
                take_ = v_ > gv
                gv, gf = jnp.where(take_, v_, gv), jnp.where(take_, i_, gf)
            gi_ = gf.astype(I32)
            tokc = jnp.where(i0 < B, gi_, 0)
            tok_m = tokc if M == XR else jnp.concatenate([tokc, jnp.zeros((M - XR, 1), I32)], axis=0)
            out_ref[...] = jnp.where(lax.broadcasted_iota(I32, (M, 128), 1) == l, tok_m, out_ref[...])

            @pl.when(l + 1 < L)
            def _():
                scr["tokv"][...] = jnp.broadcast_to(tokc, (XR, 128))
                c_ = pltpu.make_async_copy(scr["tokv"], scr["tok_s"], scr["misc_sem"])
                c_.start()
                c_.wait()
                l1 = jnp.minimum(l + 1, L - 1)
                prev = scr["carry"][...]
                if M > XR:
                    prev = jnp.concatenate([prev, jnp.zeros((M - XR, H), F32)], axis=0)
                draft_prologue(lambda t_: scr["tok_s"][t_, 0] if t_ < XR else jnp.int32(0),
                               lambda t_: rmeta_a[l1 * TMAX + t_, 0] > 0 if t_ < XR else jnp.bool_(False), prev, slot)

        # ---------------- step ----------------
        pltpu.make_async_copy(smalls.at[0], scr["sm"].at[0], scr["sm_sem"].at[0]).start()
        if draft:
            out_ref[...] = jnp.zeros((M, 128), I32)
            pltpu.make_async_copy(smalls.at[0], scr["sm"].at[0], scr["sm_sem"].at[0]).wait()
            prev0 = h0[...]
            if M > TMAX:
                prev0 = jnp.concatenate([prev0, jnp.zeros((M - TMAX, H), F32)], axis=0)
            draft_prologue(lambda t_: dmeta[3, t_], lambda t_: rmeta[t_, 0] > 0, prev0, 0)
            # layer 0 waits slot 0 again
            pltpu.make_async_copy(smalls.at[0], scr["sm"].at[0], scr["sm_sem"].at[0]).start()
        else:
            h_rows = [h0[pl.ds(me + R * k_, 1), :] for k_ in range(RPR)]
            scr["h"][...] = own_rows_pad(jnp.concatenate(h_rows, axis=0))
            scr["res"][...] = jnp.zeros((RO8, H), F32)
        scr["osend"][...] = jnp.zeros(scr["osend"].shape, U32)
        scr["own_v"][...] = jnp.full(scr["own_v"].shape, -1, I32)
        scr["own_uv"][...] = jnp.full(scr["own_uv"].shape, -1, I32)
        scr["memb_v"][...] = jnp.zeros(scr["memb_v"].shape, I32)

        def zn(t_, c):
            scr["nown_s"][t_] = jnp.int32(0)
            return c
        lax.fori_loop(0, TMAX, zn, 0)

        def zb(b_, c):
            scr["nu_s"][b_] = jnp.int32(0)
            scr["uo_s"][b_] = jnp.int32(0)
            scr["nuo_s"][b_] = jnp.int32(0)
            return c
        lax.fori_loop(0, BMAX, zb, 0)
        if "weights" in skip:
            for k in tkinds:
                scr["ibuf_" + k][...] = jnp.zeros(scr["ibuf_" + k].shape, wdtype[k])
            if len(tkinds) < len(wshape):
                scr["pool"][...] = jnp.zeros(scr["pool"].shape, U8)
        lax.fori_loop(0, L, layer, 0)
        if draft:
            return
        x = scr["h"][...] + scr["res"][...]
        pltpu.make_async_copy(smalls.at[L], scr["sm"].at[0], scr["sm_sem"].at[0]).start()
        pltpu.make_async_copy(smalls.at[L], scr["sm"].at[0], scr["sm_sem"].at[0]).wait()
        o, n = lay["in_ln"]
        y = rmsnorm(x, scr["sm"][0, :, o:o + n], cfg.eps)
        out_ref[...] = gather_rows_all(y, "ag_send", "ag_recv").astype(BF16)

    # ---------------- pallas_call ----------------
    in_specs = ([pl.BlockSpec(memory_space=pltpu.SMEM)] * 4
                + [pl.BlockSpec(memory_space=pltpu.VMEM)] * 3
                + [pl.BlockSpec(memory_space=pl.ANY)]
                + ([pl.BlockSpec(memory_space=pltpu.SMEM)] + [pl.BlockSpec(memory_space=pl.ANY)] * 3
                   if draft else [])
                + [pl.BlockSpec(memory_space=pl.ANY)] * (n_w + n_c))
    out0 = jax.ShapeDtypeStruct((M, 128), I32) if draft else jax.ShapeDtypeStruct((M, H), BF16)
    out_shape = [out0, jax.ShapeDtypeStruct((L, TMAX, 128), U32),
                 jax.ShapeDtypeStruct((L, TMAX, 128), U32), jax.ShapeDtypeStruct((L, TMAX, 256), I32)]
    out_specs = [pl.BlockSpec(memory_space=pltpu.VMEM)] + [pl.BlockSpec(memory_space=pl.ANY)] * 3
    if debug:
        out_shape += [jax.ShapeDtypeStruct((L, 2, RPR, H), F32),
                      jax.ShapeDtypeStruct((L, TMAX, KR_, 128), I32),
                      jax.ShapeDtypeStruct((L, TMAX, 128), F32),
                      jax.ShapeDtypeStruct((L, TMAX, NH, KL), F32)]
        out_specs += [pl.BlockSpec(memory_space=pl.ANY)] * 4
    call = pl.pallas_call(
        kernel,
        name="glm5_mk_draft" if draft else "glm5_mk_dcp",
        out_shape=out_shape,
        in_specs=in_specs,
        out_specs=out_specs,
        scratch_shapes=[scratch[k] for k in s_names],
        compiler_params=pltpu.CompilerParams(
            collective_id=COLLECTIVE_ID if R > 1 else None, vmem_limit_bytes=100 * 1024 * 1024,
            disable_bounds_checks=True),
        interpret=pltpu.InterpretParams() if interpret else False,
    )

    vmem_by = {k: int(np.prod(v.shape)) * jnp.dtype(v.dtype).itemsize for k, v in scratch.items()
               if str(getattr(v, "memory_space", "")).lower().endswith("vmem")}
    info = dict(s_names=s_names, w_names=w_names, c_names=c_names, tk=tk, S=S, ROWSL=ROWSL, M=M, wk=wk,
                ck=ck, vmem_scratch_bytes=sum(vmem_by.values()), vmem_by=vmem_by, stack_w=SW, stack_c=SC)
    if draft:
        info.update(VR=VR, VRP=VRP, XR=XR, DH=DH)
    return call, info


# ------------------------------------------------------------------------------------------
# The cache writer: a second, barrier-free call that applies the step's owned new rows
# ------------------------------------------------------------------------------------------

def build_writer(cfg, *, nb, max_pages, bmax=None, interpret=False, axis_name=None, stack_c=None):
    """Returns (writer_fn, info). writer_fn(scal, rmeta, bt, nrow, rrow, irec, *caches) -> caches,
    the caches aliased in -> out (no barrier: an aliased operand of a barrier-carrying call is copied
    in and out by XLA, probes/02-alias-copy). nrow / rrow / irec are the megakernel's staging outputs
    (identical on every rank); each rank writes the rows whose position it owns, for every layer:
      nope: the row's 128 words -> (page, slot), one HBM -> HBM DMA per (layer, row);
      rope: the 4-token row (page, slot >> 2) read for every layer, the quarters of every owned new
            token in that row overlaid, written back;
      idx:  the SAL page read, the owned new tokens' record columns patched, written back (per
            distinct page, in row order)."""
    R, L, TMAX = cfg.R, cfg.L, cfg.TMAX
    BMAX = bmax or TMAX
    pg = cfg.page
    lg_pg = int(math.log2(pg))
    ILV, G_ = cfg.ILV, cfg.ILV * cfg.R
    IR4 = cfg.IREC // 4
    ck = cache_kinds(cfg, nb)
    SCG = cache_groups(stack_c, ck)
    SC = frozenset(SCG)
    c_names = cache_names(ck, SCG)
    n_c = len(c_names)
    CPOS = {}
    for k, (ls, _, _) in ck.items():
        ls = list(ls)
        if ls == list(range(len(ls))):
            CPOS[k] = lambda l: l
        else:
            fr, sk = cfg.idx_freq, cfg.idx_skip
            CPOS[k] = lambda l, fr=fr, sk=sk, n=len(ls): jnp.clip(
                jnp.where(l < sk, l, sk + (l - sk) // fr), 0, n - 1)
    scratch = {
        "rold": pltpu.VMEM((L, TMAX, 128), U32), "rnew": pltpu.VMEM((L, TMAX, 128), U32),
        "ipage": pltpu.VMEM((IR4, 4, pg), U8), "rec_v": pltpu.VMEM((TMAX, 256), I32),
        "rec_s": pltpu.SMEM((TMAX, 256), I32),
        "sem": pltpu.SemaphoreType.DMA((4,)),
    }
    s_names = list(scratch)
    n_fixed = 6

    def kernel(*refs):
        scal, rmeta, bt, nrow, rrow, irec = refs[:n_fixed]
        c_out = refs[n_fixed + n_c:n_fixed + 2 * n_c]
        scr = dict(zip(s_names, refs[n_fixed + 2 * n_c:]))
        me = scal[7] if axis_name is None else lax.axis_index(axis_name).astype(I32)
        T = scal[0]
        C = {}
        CS = {}
        for (k, l), r in zip(c_names, c_out):
            if isinstance(l, str):
                CS.setdefault(k, []).append(r)
            else:
                C.setdefault(k, {})[l] = r

        def csel(k, l, fn):
            if k in SC:
                ng = len(CS[k])
                pos = CPOS[k](l)
                if ng == 1:
                    fn(CS[k][0].at[pos])
                else:
                    gs = len(ck[k][0]) // ng
                    bswitch(pos // gs, {gi: (lambda gi=gi: fn(CS[k][gi].at[pos - gi * gs]))
                                        for gi in range(ng)})
            else:
                bswitch(l, {ll: (lambda r=r: fn(r)) for ll, r in C[k].items()})

        def mine(t_):
            pos = rmeta[t_, 1]
            return jnp.logical_and(rmeta[t_, 0] > 0, (pos % G_) // ILV == me)

        def where_(t_):
            pos = rmeta[t_, 1]
            req = rmeta[t_, 2]
            lo = (pos // G_) * ILV + pos % ILV
            return bt[req, lo >> lg_pg], lo & (pg - 1), pos

        # staged rope words of every row, every layer (one DMA)
        cp = pltpu.make_async_copy(rrow, scr["rnew"], scr["sem"].at[0])
        cp.start()
        cp.wait()

        def per_layer(l, c):
            # nope rows: HBM -> HBM
            def nope(t_, c2):
                @pl.when(jnp.logical_and(t_ < T, mine(t_)))
                def _():
                    page, sl, _p = where_(t_)
                    csel("nope", l, lambda r: pltpu.make_async_copy(
                        nrow.at[l, pl.ds(t_, 1)], r.at[page, pl.ds(sl, 1)], scr["sem"].at[1]).start())
                return c2
            lax.fori_loop(0, TMAX, nope, 0)

            # rope rows: read every owned row's 4-token row
            def rread(t_, c2):
                @pl.when(jnp.logical_and(t_ < T, mine(t_)))
                def _():
                    page, sl, _p = where_(t_)
                    csel("rope", l, lambda r: pltpu.make_async_copy(
                        r.at[page, pl.ds(sl >> 2, 1)], scr["rold"].at[l, pl.ds(t_, 1)],
                        scr["sem"].at[2]).start())
                return c2
            lax.fori_loop(0, TMAX, rread, 0)

            def rwait(t_, c2):
                @pl.when(jnp.logical_and(t_ < T, mine(t_)))
                def _():
                    pltpu.make_async_copy(rrow.at[0, pl.ds(0, 1)], scr["rold"].at[0, pl.ds(0, 1)],
                                          scr["sem"].at[2]).wait()
                return c2
            lax.fori_loop(0, TMAX, rwait, 0)
            lane = lax.broadcasted_iota(I32, (1, 128), 1)

            def rmerge(t_, c2):
                @pl.when(jnp.logical_and(t_ < T, mine(t_)))
                def _():
                    page, sl, pos = where_(t_)
                    req = rmeta[t_, 2]
                    row = scr["rold"][l, pl.ds(t_, 1), :]

                    def ov(t2, row_):
                        pos2 = rmeta[t2, 1]
                        same = jnp.logical_and(jnp.logical_and(t2 < T, mine(t2)), jnp.logical_and(
                            rmeta[t2, 2] == req, pos2 // ILV == pos // ILV))
                        q2 = pos2 & 3
                        new = scr["rnew"][l, pl.ds(t2, 1), :]
                        return jnp.where(jnp.logical_and(same, lane // 32 == q2), new, row_)
                    row = lax.fori_loop(0, TMAX, ov, row)
                    scr["rold"][l, pl.ds(t_, 1), :] = row
                    csel("rope", l, lambda r: pltpu.make_async_copy(
                        scr["rold"].at[l, pl.ds(t_, 1)], r.at[page, pl.ds(sl >> 2, 1)],
                        scr["sem"].at[3]).start())
                return c2
            lax.fori_loop(0, TMAX, rmerge, 0)
            return c
        lax.fori_loop(0, L, per_layer, 0)

        # idx pages, per fresh layer
        lane_pg = lax.broadcasted_iota(I32, (4, pg), 1)
        sub4 = lax.broadcasted_iota(I32, (4, pg), 0)

        def per_idx_layer(l, c):
            full = jnp.logical_or(l < cfg.idx_skip,
                                  (l - cfg.idx_skip) % cfg.idx_freq == cfg.idx_freq - 1)

            @pl.when(full)
            def _():
                cp_ = pltpu.make_async_copy(irec.at[l], scr["rec_v"], scr["sem"].at[0])
                cp_.start()
                cp_.wait()
                cp_ = pltpu.make_async_copy(scr["rec_v"], scr["rec_s"], scr["sem"].at[0])
                cp_.start()
                cp_.wait()

                def body(t_, cur):
                    ok = jnp.logical_and(t_ < T, mine(t_))
                    page, sl, _p = where_(t_)

                    @pl.when(ok)
                    def _():
                        def go(r):
                            @pl.when(jnp.logical_and(page != cur, cur >= 0))
                            def _():
                                c3 = pltpu.make_async_copy(scr["ipage"], r.at[cur], scr["sem"].at[0])
                                c3.start()
                                c3.wait()

                            @pl.when(page != cur)
                            def _():
                                c3 = pltpu.make_async_copy(r.at[page], scr["ipage"], scr["sem"].at[0])
                                c3.start()
                                c3.wait()
                            for w in range(IR4):
                                v4 = [scr["rec_s"][t_, 4 * w + q_] for q_ in range(4)]
                                col = jnp.where(sub4 == 0, v4[0], jnp.where(sub4 == 1, v4[1], jnp.where(
                                    sub4 == 2, v4[2], v4[3])))
                                old = scr["ipage"][w].astype(I32)
                                scr["ipage"][w] = jnp.where(lane_pg == sl, col, old).astype(U8)
                        csel("idx", l, go)
                    return jnp.where(ok, page, cur)
                cur = lax.fori_loop(0, TMAX, body, jnp.int32(-1))

                @pl.when(cur >= 0)
                def _():
                    csel("idx", l, lambda r: (lambda c3: (c3.start(), c3.wait()))(
                        pltpu.make_async_copy(scr["ipage"], r.at[cur], scr["sem"].at[0])))
            return c
        lax.fori_loop(0, L, per_idx_layer, 0)

        # drain the nope and rope writes (one wait per issued row-DMA, same bytes each)
        n_own = jnp.int32(0)
        for t_ in range(TMAX):
            n_own = n_own + jnp.logical_and(t_ < T, mine(t_)).astype(I32)

        def drain(i, c):
            pltpu.make_async_copy(nrow.at[0, pl.ds(0, 1)], scr["rold"].at[0, pl.ds(0, 1)],
                                  scr["sem"].at[1]).wait()
            pltpu.make_async_copy(nrow.at[0, pl.ds(0, 1)], scr["rold"].at[0, pl.ds(0, 1)],
                                  scr["sem"].at[3]).wait()
            return c
        lax.fori_loop(0, n_own * L, drain, 0)

    c_shapes = [jax.ShapeDtypeStruct(((len(ck[k][0]) // SCG[k],) if isinstance(l, str) else ())
                                     + ck[k][1], ck[k][2]) for k, l in c_names]
    call = pl.pallas_call(
        kernel,
        name="glm5_mk_dcp_writer",
        out_shape=c_shapes,
        in_specs=([pl.BlockSpec(memory_space=pltpu.SMEM)] * 3 + [pl.BlockSpec(memory_space=pl.ANY)] * 3
                  + [pl.BlockSpec(memory_space=pl.ANY)] * n_c),
        out_specs=[pl.BlockSpec(memory_space=pl.ANY)] * n_c,
        scratch_shapes=[scratch[k] for k in s_names],
        input_output_aliases={n_fixed + i: i for i in range(n_c)},
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=64 * 1024 * 1024,
                                             disable_bounds_checks=True),
        interpret=pltpu.InterpretParams() if interpret else False,
    )
    return call, dict(c_names=c_names, ck=ck)
