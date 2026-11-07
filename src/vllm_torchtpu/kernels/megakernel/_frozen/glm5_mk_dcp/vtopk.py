"""Vectorized top-k extraction (v11): selected keys -> ascending token ids with no scalar loop.

The threshold search leaves T (the K-th largest key) and the selection mask
    sel = (key > T) | (key == T & pos < P)
where P keeps only the first K - #(key > T) ties in position order (a 19-pass position search that
runs only when ties exceed the need). Output slot k (ascending) then lives in the row r whose
selected-count prefix brackets it, CE[r] <= k < CI[r], at the (k - CE[r])-th selected lane of r.
Everything is counting: per-row counts and their prefix sums, and one one-hot matmul per k chunk
that reads each slot's row and within-row prefix back. Every value is a small integer: exact in bf16
(<= 256) with f32 accumulation, hi/lo split where larger.
"""
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl

F32, I32, BF16 = jnp.float32, jnp.int32, jnp.bfloat16
NT = (((1,), (1,)), ((), ()))
NN = (((1,), (0,)), ((), ()))
INT_MIN = -(2 ** 31)
INT_MAX = 2 ** 31 - 1


def shl(x, n):
    return lax.shift_left(x, jnp.asarray(n, x.dtype))


def threshold(keys_ref, K):
    """2-bit radix threshold search (v8): the K-th largest key, as a biased i32."""
    def pass_(i, p):
        sh = jnp.int32(30) - 2 * i
        kk_ = keys_ref[...]
        dig = jnp.int32(0)
        for k in (1, 2, 3):
            cs = (p | shl(jnp.int32(k), sh)) ^ jnp.int32(INT_MIN)
            dig = dig + (jnp.sum((kk_ >= cs).astype(I32)) >= K).astype(I32)
        return p | shl(dig, sh)
    p = lax.fori_loop(0, 16, pass_, jnp.int32(0))
    return p ^ jnp.int32(INT_MIN)


def extract(keys_ref, T, K, idx_v, kc=256):
    """keys_ref (ROWS, 128) i32, T from threshold() -> idx_v (K // 128, 128) i32 ascending token ids,
    -1 past the selected count. Returns the selected count."""
    ROWS = keys_ref.shape[0]
    assert ROWS % 128 == 0 and K % 128 == 0 and K <= 2 ** 15 and ROWS <= 128 * 128
    NBK = ROWS // 128
    kc = min(kc, K)
    assert K % kc == 0
    kk = keys_ref[...]
    gt = kk > T
    eq = kk == T
    n_gt = jnp.sum(gt.astype(I32))
    n_eq = jnp.sum(eq.astype(I32))
    need_eq = jnp.where(T == jnp.int32(INT_MIN), 0, K - n_gt)
    pos = (lax.broadcasted_iota(I32, (ROWS, 128), 0) * 128
           + lax.broadcasted_iota(I32, (ROWS, 128), 1))
    excess = n_eq > need_eq
    PB = (ROWS * 128).bit_length()

    def pbit(i, p):
        cand = p | shl(jnp.int32(1), jnp.int32(PB - 1) - i)
        c = jnp.sum((eq & (pos < cand)).astype(I32))
        return jnp.where(c <= need_eq, cand, p)
    P = lax.fori_loop(0, jnp.where(excess, PB, 0), pbit, jnp.int32(0))
    P = jnp.where(excess, P, jnp.int32(INT_MAX))
    sel = gt | (eq & (pos < P))
    n_sel = jnp.sum(sel.astype(I32))
    selb = sel.astype(F32).astype(BF16)                                   # (ROWS, 128)

    i128 = lax.broadcasted_iota(I32, (128, 128), 0)
    j128 = lax.broadcasted_iota(I32, (128, 128), 1)
    # per-row counts along lanes, blocked: C[b, j] = count of row 128 b + j
    cnt = lax.dot_general(jnp.ones((8, 128), BF16), selb, NT, preferred_element_type=F32)
    C = jnp.concatenate([cnt[0:1, b * 128:(b + 1) * 128] for b in range(NBK)]
                        + [jnp.zeros((128 - NBK, 128), F32)], axis=0)       # (128, 128)
    W = lax.dot_general(C.astype(BF16), (i128 <= j128).astype(BF16), NN,
                        preferred_element_type=F32)                          # within-block incl
    tot = W[:, 127:128]
    th = jnp.floor(tot * (1.0 / 128))
    tl = tot - 128 * th
    Ls = (j128 < i128).astype(BF16)                                          # Ls[b, b'] = b' < b
    off = (128 * lax.dot_general(Ls, jnp.broadcast_to(th, (128, 128)).astype(BF16), NN,
                                 preferred_element_type=F32)
           + lax.dot_general(Ls, jnp.broadcast_to(tl, (128, 128)).astype(BF16), NN,
                             preferred_element_type=F32))
    CI = W + off                                                             # row-count prefix, incl
    CE = CI - C
    CEt, CIt = CE.T, CI.T                                                    # [j, b] = row 128 b + j
    ce_col = jnp.concatenate([CEt[:, b:b + 1] for b in range(NBK)], axis=0)  # (ROWS, 1)
    ci_col = jnp.concatenate([CIt[:, b:b + 1] for b in range(NBK)], axis=0)
    ce_row = jnp.concatenate([CE[b:b + 1, :] for b in range(NBK)], axis=1)   # (1, ROWS)

    # VT: within-row inclusive prefix (lane l, row r) and each row's id / CE, hi-lo split
    wrt = lax.dot_general((j128 <= i128).astype(BF16), selb, NT, preferred_element_type=F32)
    r = lax.broadcasted_iota(I32, (1, ROWS), 1).astype(F32)
    rh = jnp.floor(r * (1.0 / 128))
    ceh = jnp.floor(ce_row * (1.0 / 128))
    tail = jnp.concatenate([rh, r - 128 * rh, ceh, ce_row - 128 * ceh,
                            jnp.zeros((12, ROWS), F32)], axis=0)
    vt = jnp.concatenate([wrt.astype(BF16), tail.astype(BF16)], axis=0)      # (144, ROWS)

    for c0 in range(0, K, kc):
        kr = (c0 + lax.broadcasted_iota(I32, (1, kc), 1)).astype(F32)
        oht = ((ce_col <= kr) & (kr < ci_col)).astype(F32).astype(BF16)      # (ROWS, kc)
        R = lax.dot_general(vt, oht, NN, preferred_element_type=F32)          # (144, kc)
        rr = 128 * R[128:129] + R[129:130]
        ce = 128 * R[130:131] + R[131:132]
        lane = jnp.sum((R[0:128] <= kr - ce).astype(F32), axis=0, keepdims=True)
        tok = jnp.where(kr < n_sel.astype(F32), rr * 128 + lane, -1.0).astype(I32)
        for q in range(kc // 128):
            idx_v[c0 // 128 + q:c0 // 128 + q + 1, :] = tok[:, q * 128:(q + 1) * 128]
    return n_sel


def extract_blk(keys_ref, T, K, idx_v, vt_ref, acc_ref, cc_ref, kc=256):
    """probe 27 tb: extract() with the same counting math, 128-row block by block. Output slot k
    lives in the row r with CE[r] <= k < CI[r]; CE / CI are non-decreasing in r, so the rows a k
    chunk [c0, c0 + kc) can read form one contiguous range, found by two counts. Each chunk's
    one-hot matmul then runs over only the 128-row blocks of that range (<= NBK + #chunks block
    matmuls in total, against NBK x #chunks) and nothing ROWS-sized is ever a live value.
    vt_ref (ROWS, 256) bf16: row r = [within-row inclusive prefix (128) | r_hi, r_lo, CE_hi, CE_lo].
    acc_ref (kc, 256) f32, cc_ref (2, 128, 128) f32 (CE, CI in [block, row-in-block] layout).
    Bit-identical to extract(): every one-hot column selects exactly one row (sums with zeros)."""
    ROWS = keys_ref.shape[0]
    assert ROWS % 128 == 0 and K % 128 == 0 and K <= 2 ** 15 and ROWS <= 128 * 128
    NBK = ROWS // 128
    kc = min(kc, K)
    assert K % kc == 0
    ri = lax.broadcasted_iota(I32, (128, 128), 0)
    li = lax.broadcasted_iota(I32, (128, 128), 1)

    def blk(b):
        return keys_ref[b * 128:(b + 1) * 128, :]

    def posb(b):
        return (b * 128 + ri) * 128 + li

    cgt = jnp.zeros((128, 128), I32)
    ceq = jnp.zeros((128, 128), I32)
    for b in range(NBK):
        kb = blk(b)
        cgt = cgt + (kb > T).astype(I32)
        ceq = ceq + (kb == T).astype(I32)
    n_gt = jnp.sum(cgt)
    n_eq = jnp.sum(ceq)
    need_eq = jnp.where(T == jnp.int32(INT_MIN), 0, K - n_gt)
    excess = n_eq > need_eq
    PB = (ROWS * 128).bit_length()

    def pbit(i, p):
        cand = p | shl(jnp.int32(1), jnp.int32(PB - 1) - i)
        cv = jnp.zeros((128, 128), I32)
        for b in range(NBK):
            cv = cv + ((blk(b) == T) & (posb(b) < cand)).astype(I32)
        return jnp.where(jnp.sum(cv) <= need_eq, cand, p)
    P = lax.fori_loop(0, jnp.where(excess, PB, 0), pbit, jnp.int32(0))
    P = jnp.where(excess, P, jnp.int32(INT_MAX))

    ones8 = jnp.ones((8, 128), BF16)
    upper = (ri <= li).astype(BF16)                                          # [l', l] = l' <= l
    rows_c = []
    for b in range(NBK):
        kb = blk(b)
        selb = ((kb > T) | ((kb == T) & (posb(b) < P))).astype(F32).astype(BF16)
        rows_c.append(lax.dot_general(ones8, selb, NT, preferred_element_type=F32)[0:1])
        vt_ref[b * 128:(b + 1) * 128, 0:128] = lax.dot_general(
            selb, upper, NN, preferred_element_type=F32).astype(BF16)          # within-row incl
    C = jnp.concatenate(rows_c + [jnp.zeros((128 - NBK, 128), F32)], axis=0)  # [b, j]
    n_sel = jnp.sum(C).astype(I32)
    W = lax.dot_general(C.astype(BF16), upper, NN, preferred_element_type=F32)
    tot = W[:, 127:128]
    th = jnp.floor(tot * (1.0 / 128))
    tl = tot - 128 * th
    Ls = (li < ri).astype(BF16)                                              # Ls[b, b'] = b' < b
    off = (128 * lax.dot_general(Ls, jnp.broadcast_to(th, (128, 128)).astype(BF16), NN,
                                 preferred_element_type=F32)
           + lax.dot_general(Ls, jnp.broadcast_to(tl, (128, 128)).astype(BF16), NN,
                             preferred_element_type=F32))
    CI = W + off
    CE = CI - C
    cc_ref[0] = CE
    cc_ref[1] = CI
    CEt = CE.T                                                               # [j, b]
    for b in range(NBK):
        ce_c = CEt[:, b:b + 1]
        ceh = jnp.floor(ce_c * (1.0 / 128))
        tail = jnp.where(li == 0, float(b), jnp.where(li == 1, ri.astype(F32), jnp.where(
            li == 2, ceh, jnp.where(li == 3, ce_c - 128 * ceh, 0.0))))
        vt_ref[b * 128:(b + 1) * 128, 128:256] = tail.astype(BF16)

    valid = ri < NBK
    for c0 in range(0, K, kc):
        kr = (c0 + lax.broadcasted_iota(I32, (kc, 1), 0)).astype(F32)       # (kc, 1)
        r_first = jnp.sum(((CI <= c0) & valid).astype(I32))
        r_end = jnp.sum(((CE < c0 + kc) & valid).astype(I32))
        b_lo = r_first // 128
        b_hi = jnp.maximum((r_end + 127) // 128, b_lo)
        acc_ref[...] = jnp.zeros(acc_ref.shape, F32)

        def body(b, c):
            ce_r = cc_ref[0, pl.ds(b, 1), :]                                 # (1, 128)
            ci_r = cc_ref[1, pl.ds(b, 1), :]
            oht = ((ce_r <= kr) & (kr < ci_r)).astype(F32).astype(BF16)      # (kc, 128)
            r0 = pl.multiple_of(b * 128, 128)
            acc_ref[...] += lax.dot_general(oht, vt_ref[pl.ds(r0, 128), :], NN,
                                            preferred_element_type=F32)
            return c
        lax.fori_loop(b_lo, b_hi, body, 0)
        RT = acc_ref[...]                                                    # (kc, 256)
        rr = 128 * RT[:, 128:129] + RT[:, 129:130]
        ce = 128 * RT[:, 130:131] + RT[:, 131:132]
        lane = jnp.sum((RT[:, 0:128] <= kr - ce).astype(F32), axis=1, keepdims=True)
        tok = jnp.where(kr < n_sel.astype(F32), rr * 128 + lane, -1.0)       # (kc, 1)
        tokr = jnp.transpose(jnp.broadcast_to(tok, (kc, 128)))[0:1].astype(I32)   # (1, kc)
        for q in range(kc // 128):
            idx_v[c0 // 128 + q:c0 // 128 + q + 1, :] = tokr[:, q * 128:(q + 1) * 128]
    return n_sel


def _blk(ref, b):
    return ref[pl.ds(pl.multiple_of(b * 128, 128), 128), :]


def _cnt8(m):
    """(128, 128) bool -> (8, 128) i32 partial counts (vreg adds only)."""
    return jnp.sum(m.astype(I32).reshape(16, 8, 128), axis=0)


def threshold_roll(keys_ref, K, seq_len, unroll=1):
    """The kernel's slow-path key mask (positions >= seq_len -> INT_MIN, in place) and threshold()'s
    16 2-bit radix passes, each a fori loop over 128-row blocks: same counts, code independent of
    ROWS. Returns T (the K-th largest key)."""
    NBK = keys_ref.shape[0] // 128
    ri = lax.broadcasted_iota(I32, (128, 128), 0)
    li = lax.broadcasted_iota(I32, (128, 128), 1)

    def mask(b, c):
        r0 = pl.multiple_of(b * 128, 128)
        pos = (b * 128 + ri) * 128 + li
        keys_ref[pl.ds(r0, 128), :] = jnp.where(pos >= seq_len, jnp.int32(INT_MIN),
                                                keys_ref[pl.ds(r0, 128), :])
        return c
    lax.fori_loop(0, NBK, mask, 0, unroll=min(unroll, NBK))

    def pass_(i, p):
        sh = jnp.int32(30) - 2 * i
        cs = [(p | shl(jnp.int32(k), sh)) ^ jnp.int32(INT_MIN) for k in (1, 2, 3)]

        def body(b, acc):
            kb = _blk(keys_ref, b)
            return tuple(a + _cnt8(kb >= c) for a, c in zip(acc, cs))
        z = jnp.zeros((8, 128), I32)
        acc = lax.fori_loop(0, NBK, body, (z, z, z), unroll=min(unroll, NBK))
        dig = jnp.int32(0)
        for a in acc:
            dig = dig + (jnp.sum(a) >= K).astype(I32)
        return p | shl(dig, sh)
    p = lax.fori_loop(0, 16, pass_, jnp.int32(0))
    return p ^ jnp.int32(INT_MIN)


def extract_roll(keys_ref, T, K, idx_v, acc_ref, cc_ref, kc=256, unroll=1):
    """extract_blk() with every per-block loop a fori loop and no (ROWS, 256) scratch: once a
    block's selection is known, its within-row inclusive prefix (<= 128, exact) is written over its
    keys, and each row's (r_hi, r_lo, CE_hi, CE_lo) tail is rebuilt from cc_ref in the chunk loop.
    The prefix and tail columns are separate matmuls (independent output columns), so the result is
    bit-identical to extract() / extract_blk(). keys_ref is consumed (the kernel rewrites every row
    before the next read). acc_ref (kc, 256) f32, cc_ref (2, 128, 128) f32."""
    ROWS = keys_ref.shape[0]
    assert ROWS % 128 == 0 and K % 128 == 0 and K <= 2 ** 15 and ROWS <= 128 * 128
    NBK = ROWS // 128
    kc = min(kc, K)
    assert K % kc == 0
    ri = lax.broadcasted_iota(I32, (128, 128), 0)
    li = lax.broadcasted_iota(I32, (128, 128), 1)

    def posb(b):
        return (b * 128 + ri) * 128 + li

    def cnt(b, c):
        kb = _blk(keys_ref, b)
        return c[0] + _cnt8(kb > T), c[1] + _cnt8(kb == T)
    z = jnp.zeros((8, 128), I32)
    cgt, ceq = lax.fori_loop(0, NBK, cnt, (z, z), unroll=min(unroll, NBK))
    n_gt = jnp.sum(cgt)
    n_eq = jnp.sum(ceq)
    need_eq = jnp.where(T == jnp.int32(INT_MIN), 0, K - n_gt)
    excess = n_eq > need_eq
    PB = (ROWS * 128).bit_length()

    def pbit(i, p):
        cand = p | shl(jnp.int32(1), jnp.int32(PB - 1) - i)

        def body(b, cv):
            return cv + _cnt8((_blk(keys_ref, b) == T) & (posb(b) < cand))
        cv = lax.fori_loop(0, NBK, body, jnp.zeros((8, 128), I32),
                           unroll=min(unroll, NBK))
        return jnp.where(jnp.sum(cv) <= need_eq, cand, p)
    P = lax.fori_loop(0, jnp.where(excess, PB, 0), pbit, jnp.int32(0))
    P = jnp.where(excess, P, jnp.int32(INT_MAX))

    ones8 = jnp.ones((8, 128), BF16)
    upper = (ri <= li).astype(BF16)                                          # [l', l] = l' <= l
    cc_ref[0] = jnp.zeros((128, 128), F32)

    def sel_blk(b, c):
        r0 = pl.multiple_of(b * 128, 128)
        kb = keys_ref[pl.ds(r0, 128), :]
        selb = ((kb > T) | ((kb == T) & (posb(b) < P))).astype(F32).astype(BF16)
        cc_ref[0, pl.ds(b, 1), :] = lax.dot_general(ones8, selb, NT, preferred_element_type=F32)[0:1]
        keys_ref[pl.ds(r0, 128), :] = lax.dot_general(
            selb, upper, NN, preferred_element_type=F32).astype(I32)             # within-row incl
        return c
    lax.fori_loop(0, NBK, sel_blk, 0, unroll=min(unroll, NBK))
    C = cc_ref[0]                                                            # [b, j]
    n_sel = jnp.sum(C).astype(I32)
    W = lax.dot_general(C.astype(BF16), upper, NN, preferred_element_type=F32)
    tot = W[:, 127:128]
    th = jnp.floor(tot * (1.0 / 128))
    tl = tot - 128 * th
    Ls = (li < ri).astype(BF16)                                              # Ls[b, b'] = b' < b
    off = (128 * lax.dot_general(Ls, jnp.broadcast_to(th, (128, 128)).astype(BF16), NN,
                                 preferred_element_type=F32)
           + lax.dot_general(Ls, jnp.broadcast_to(tl, (128, 128)).astype(BF16), NN,
                             preferred_element_type=F32))
    CI = W + off
    CE = CI - C
    cc_ref[0] = CE
    cc_ref[1] = CI

    valid = ri < NBK

    def chunk(ci, c):                     # k chunk [c0, c0 + kc): a loop too (K // kc bodies -> 1)
        c0 = ci * kc
        kr = (c0 + lax.broadcasted_iota(I32, (kc, 1), 0)).astype(F32)       # (kc, 1)
        r_first = jnp.sum(((cc_ref[1] <= c0) & valid).astype(I32))       # CI, CE re-read: not
        r_end = jnp.sum(((cc_ref[0] < c0 + kc) & valid).astype(I32))     # live across the loop
        b_lo = r_first // 128
        b_hi = jnp.maximum((r_end + 127) // 128, b_lo)
        acc_ref[...] = jnp.zeros(acc_ref.shape, F32)

        def body(b, c):
            ce_r = cc_ref[0, pl.ds(b, 1), :]                                 # (1, 128)
            ci_r = cc_ref[1, pl.ds(b, 1), :]
            oht = ((ce_r <= kr) & (kr < ci_r)).astype(F32).astype(BF16)      # (kc, 128)
            r0 = pl.multiple_of(b * 128, 128)
            pre = keys_ref[pl.ds(r0, 128), :].astype(F32).astype(BF16)
            ce_c = jnp.transpose(jnp.broadcast_to(ce_r, (128, 128)))         # [r, *] = CE[b, r]
            ceh = jnp.floor(ce_c * (1.0 / 128))
            tail = jnp.where(li == 0, b.astype(F32), jnp.where(li == 1, ri.astype(F32), jnp.where(
                li == 2, ceh, jnp.where(li == 3, ce_c - 128 * ceh, 0.0)))).astype(BF16)
            acc_ref[:, 0:128] += lax.dot_general(oht, pre, NN, preferred_element_type=F32)
            acc_ref[:, 128:256] += lax.dot_general(oht, tail, NN, preferred_element_type=F32)
            return c
        lax.fori_loop(b_lo, b_hi, body, 0)
        RT = acc_ref[...]                                                    # (kc, 256)
        rr = 128 * RT[:, 128:129] + RT[:, 129:130]
        ce = 128 * RT[:, 130:131] + RT[:, 131:132]
        lane = jnp.sum((RT[:, 0:128] <= kr - ce).astype(F32), axis=1, keepdims=True)
        tok = jnp.where(kr < n_sel.astype(F32), rr * 128 + lane, -1.0)       # (kc, 1)
        tokr = jnp.transpose(jnp.broadcast_to(tok, (kc, 128)))[0:1].astype(I32)   # (1, kc)
        for q in range(kc // 128):
            idx_v[pl.ds(ci * (kc // 128) + q, 1), :] = tokr[:, q * 128:(q + 1) * 128]
        return c
    lax.fori_loop(0, K // kc, chunk, 0)
    return n_sel
