"""Two-level exact top-k for DCP (decode megakernel, TP x DCP).

Main's semantics (kernel_work/glm52-mk-tp16dcp16/incumbent/main-dcp-semantics.md §C):
  1. each rank selects its local top-K of the monotone i32 keys, ties -> lowest local index, output
     in ascending local-index order ("candidate slots"), -1 / -inf padded;
  2. the R candidate rows are all-gathered rank-major ([R*K]), and the global top-K over them breaks
     ties by gathered column = (lower rank, then lower candidate slot);
  3. each rank keeps the global winners that are its own candidates, as ascending local indices.

Everything here is counting on the TensorCore, reusing vtopk's radix threshold and its counting
extraction (prefix counts + one one-hot matmul per output chunk). `extract_payload` additionally reads
the selected keys' bytes through the same one-hot matmul, so the candidate keys come out in slot order
next to their indices without any vector gather (which cannot cross a vreg, hardware-facts §8).

Key buffers are (ROWS, 128) i32 with position = row * 128 + lane; INT_MIN marks an invalid entry
(every finite score maps to a key > INT_MIN). Block loops run over a traced number of 128-row blocks,
so the cost follows the live length, not the buffer.
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


def shr(x, n):
    return lax.shift_right_logical(x, jnp.asarray(n, x.dtype))


def _blk(ref, b):
    return ref[pl.ds(pl.multiple_of(b * 128, 128), 128), :]


def _cnt8(m):
    """(128, 128) bool -> (8, 128) i32 partial counts (vreg adds only)."""
    return jnp.sum(m.astype(I32).reshape(16, 8, 128), axis=0)


def mask_tail(keys_ref, nbk, n_valid):
    """positions >= n_valid -> INT_MIN, over the first nbk blocks (traced)."""
    ri = lax.broadcasted_iota(I32, (128, 128), 0)
    li = lax.broadcasted_iota(I32, (128, 128), 1)

    def body(b, c):
        r0 = pl.multiple_of(b * 128, 128)
        pos = (b * 128 + ri) * 128 + li
        keys_ref[pl.ds(r0, 128), :] = jnp.where(pos >= n_valid, jnp.int32(INT_MIN),
                                                keys_ref[pl.ds(r0, 128), :])
        return c
    lax.fori_loop(0, nbk, body, 0)


def threshold(keys_ref, nbk, K):
    """The K-th largest key over the first nbk 128-row blocks: 16 2-bit radix passes (vtopk)."""
    def pass_(i, p):
        sh = jnp.int32(30) - 2 * i
        cs = [(p | shl(jnp.int32(k), sh)) ^ jnp.int32(INT_MIN) for k in (1, 2, 3)]

        def body(b, acc):
            kb = _blk(keys_ref, b)
            return tuple(a + _cnt8(kb >= c) for a, c in zip(acc, cs))
        z = jnp.zeros((8, 128), I32)
        acc = lax.fori_loop(0, nbk, body, (z, z, z))
        dig = jnp.int32(0)
        for a in acc:
            dig = dig + (jnp.sum(a) >= K).astype(I32)
        return p | shl(dig, sh)
    p = lax.fori_loop(0, 16, pass_, jnp.int32(0))
    return p ^ jnp.int32(INT_MIN)


def tie_bound(keys_ref, nbk, T, K):
    """P such that sel = (key > T) | (key == T & pos < P) selects exactly min(K, #valid) entries,
    ties in ascending position order (the lowest positions win)."""
    ri = lax.broadcasted_iota(I32, (128, 128), 0)
    li = lax.broadcasted_iota(I32, (128, 128), 1)

    def cnt(b, c):
        kb = _blk(keys_ref, b)
        return c[0] + _cnt8(kb > T), c[1] + _cnt8(kb == T)
    z = jnp.zeros((8, 128), I32)
    cgt, ceq = lax.fori_loop(0, nbk, cnt, (z, z))
    n_gt = jnp.sum(cgt)
    n_eq = jnp.sum(ceq)
    need_eq = jnp.where(T == jnp.int32(INT_MIN), 0, K - n_gt)
    excess = n_eq > need_eq
    PB = 24                                    # positions < 2^24 (ROWS <= 128 * 1024)

    def pbit(i, p):
        cand = p | shl(jnp.int32(1), jnp.int32(PB - 1) - i)

        def body(b, cv):
            pos = (b * 128 + ri) * 128 + li
            return cv + _cnt8((_blk(keys_ref, b) == T) & (pos < cand))
        cv = lax.fori_loop(0, nbk, body, jnp.zeros((8, 128), I32))
        return jnp.where(jnp.sum(cv) <= need_eq, cand, p)
    P = lax.fori_loop(0, jnp.where(excess, PB, 0), pbit, jnp.int32(0))
    return jnp.where(excess, P, jnp.int32(INT_MAX))


def extract_payload(keys_ref, nbk, T, P, K, idx_v, key_v, pre_ref, acc_ref, cc_ref, kc=256):
    """key_v None: positions only (compaction). Selected entries sel = (key > T) | (key == T & pos < P) over the first nbk blocks ->
    idx_v (K//128, 128) i32 ascending positions (-1 past the count) and key_v (K//128, 128) i32 their
    keys (INT_MIN past the count). keys_ref is read-only here.
    pre_ref (ROWS, 128) i32 scratch, acc_ref (kc, 768) f32, cc_ref (2, 128, 128) f32.
    Returns the selected count.

    vtopk.extract_roll's counting math (bit-identical positions), plus the selected row's key bytes
    (4 byte planes, exact small integers in bf16) read through the same one-hot matmul; the lane is
    then picked from the 128 byte columns with a one-hot compare."""
    assert K % 128 == 0 and K <= 2 ** 15
    kc = min(kc, K)
    assert K % kc == 0
    ri = lax.broadcasted_iota(I32, (128, 128), 0)
    li = lax.broadcasted_iota(I32, (128, 128), 1)
    ones8 = jnp.ones((8, 128), BF16)
    upper = (ri <= li).astype(BF16)                                          # [l', l] = l' <= l
    cc_ref[0] = jnp.zeros((128, 128), F32)

    def sel_blk(b, c):
        r0 = pl.multiple_of(b * 128, 128)
        kb = keys_ref[pl.ds(r0, 128), :]
        pos = (b * 128 + ri) * 128 + li
        selb = ((kb > T) | ((kb == T) & (pos < P))).astype(F32).astype(BF16)
        cc_ref[0, pl.ds(b, 1), :] = lax.dot_general(ones8, selb, NT, preferred_element_type=F32)[0:1]
        pre_ref[pl.ds(r0, 128), :] = lax.dot_general(
            selb, upper, NN, preferred_element_type=F32).astype(I32)              # within-row incl
        return c
    lax.fori_loop(0, nbk, sel_blk, 0)
    C = cc_ref[0]                                                            # [b, j], 0 past nbk
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
    valid = ri < nbk

    def chunk(ci, c):
        c0 = ci * kc
        kr = (c0 + lax.broadcasted_iota(I32, (kc, 1), 0)).astype(F32)       # (kc, 1)
        r_first = jnp.sum(((cc_ref[1] <= c0) & valid).astype(I32))
        r_end = jnp.sum(((cc_ref[0] < c0 + kc) & valid).astype(I32))
        b_lo = r_first // 128
        b_hi = jnp.minimum(jnp.maximum((r_end + 127) // 128, b_lo), nbk)
        acc_ref[...] = jnp.zeros(acc_ref.shape, F32)

        def body(b, c):
            ce_r = cc_ref[0, pl.ds(b, 1), :]                                 # (1, 128)
            ci_r = cc_ref[1, pl.ds(b, 1), :]
            oht = ((ce_r <= kr) & (kr < ci_r)).astype(F32).astype(BF16)      # (kc, 128)
            r0 = pl.multiple_of(b * 128, 128)
            pre = pre_ref[pl.ds(r0, 128), :].astype(F32).astype(BF16)
            ce_c = jnp.transpose(jnp.broadcast_to(ce_r, (128, 128)))         # [r, *] = CE[b, r]
            ceh = jnp.floor(ce_c * (1.0 / 128))
            tail = jnp.where(li == 0, b.astype(F32), jnp.where(li == 1, ri.astype(F32), jnp.where(
                li == 2, ceh, jnp.where(li == 3, ce_c - 128 * ceh, 0.0)))).astype(BF16)
            acc_ref[:, 0:128] += lax.dot_general(oht, pre, NN, preferred_element_type=F32)
            acc_ref[:, 128:256] += lax.dot_general(oht, tail, NN, preferred_element_type=F32)
            if key_v is None:
                return c
            kb = keys_ref[pl.ds(r0, 128), :]
            for q in range(4):                                               # key byte planes
                plane = (shr(kb, 8 * q) & 0xFF).astype(F32).astype(BF16)
                acc_ref[:, 256 + 128 * q:384 + 128 * q] += lax.dot_general(
                    oht, plane, NN, preferred_element_type=F32)
            return c
        lax.fori_loop(b_lo, b_hi, body, 0)
        RT = acc_ref[...]                                                    # (kc, 768)
        rr = 128 * RT[:, 128:129] + RT[:, 129:130]
        ce = 128 * RT[:, 130:131] + RT[:, 131:132]
        lane = jnp.sum((RT[:, 0:128] <= kr - ce).astype(F32), axis=1, keepdims=True)   # (kc, 1)
        live = kr < n_sel.astype(F32)
        tok = jnp.where(live, rr * 128 + lane, -1.0)                         # (kc, 1)
        tokr = jnp.transpose(jnp.broadcast_to(tok, (kc, 128)))[0:1].astype(I32)   # (1, kc)
        for q in range(kc // 128):
            idx_v[pl.ds(ci * (kc // 128) + q, 1), :] = tokr[:, q * 128:(q + 1) * 128]
        if key_v is None:
            return c
        lsel = lax.broadcasted_iota(I32, (kc, 128), 1).astype(F32) == lane   # (kc, 128)
        key = jnp.zeros((kc, 1), I32)
        for q in range(4):
            byte = jnp.sum(jnp.where(lsel, RT[:, 256 + 128 * q:384 + 128 * q], 0.0), axis=1,
                           keepdims=True).astype(I32)
            key = key | shl(byte, 8 * q)
        key = jnp.where(live, key, jnp.int32(INT_MIN))
        keyr = jnp.transpose(jnp.broadcast_to(key, (kc, 128)))[0:1]
        for q in range(kc // 128):
            key_v[pl.ds(ci * (kc // 128) + q, 1), :] = keyr[:, q * 128:(q + 1) * 128]
        return c
    # only the chunks that hold selected entries (the global compaction owns ~K/R of K slots);
    # the rest of the outputs are prefilled
    idx_v[...] = jnp.full(idx_v.shape, -1, I32)
    if key_v is not None:
        key_v[...] = jnp.full(key_v.shape, jnp.int32(INT_MIN), I32)
    lax.fori_loop(0, (n_sel + kc - 1) // kc, chunk, 0)
    return n_sel


def local_topk(keys_ref, n_valid, K, idx_v, key_v, pre_ref, acc_ref, cc_ref, kc=256):
    """Local top-K of keys_ref[:, :] restricted to positions < n_valid (others are forced to
    INT_MIN). Writes candidate indices (ascending) and keys in slot order. Returns the count."""
    KR = K // 128
    li = lax.broadcasted_iota(I32, (KR, 128), 1)
    ri = lax.broadcasted_iota(I32, (KR, 128), 0)
    nbk = (n_valid + 128 * 128 - 1) // (128 * 128)

    @pl.when(n_valid <= K)
    def _fast():                       # every valid entry is a candidate, already in index order
        pos = ri * 128 + li
        idx_v[...] = jnp.where(pos < n_valid, pos, -1)
        key_v[...] = jnp.where(pos < n_valid, keys_ref[0:KR, :], jnp.int32(INT_MIN))

    @pl.when(n_valid > K)
    def _slow():
        mask_tail(keys_ref, nbk, n_valid)
        T = threshold(keys_ref, nbk, K)
        P = tie_bound(keys_ref, nbk, T, K)
        extract_payload(keys_ref, nbk, T, P, K, idx_v, key_v, pre_ref, acc_ref, cc_ref, kc=kc)
    return jnp.minimum(n_valid, K)


def global_owned(kall_ref, me, K, own_v, pre_ref, acc_ref, cc_ref, mk_ref, kc=256, pay_ref=None,
                 pay_out=None):
    """kall_ref (R*K//128, 128) i32: the R ranks' candidate keys, rank-major (position = r*K + slot).
    The global top-K (ties -> lower position) restricted to rank `me`'s rows, compacted:
    own_v (K//128, 128) i32 = this rank's winning candidate SLOTS ascending, -1 padded.
    mk_ref (128, 128) i32 scratch. Returns the count of owned winners."""
    KR = K // 128
    NR = kall_ref.shape[0]
    assert NR % 128 == 0
    nbk = NR // 128
    T = threshold(kall_ref, nbk, K)
    P = tie_bound(kall_ref, nbk, T, K)
    # this rank's block of KR rows: selection mask -> keys 1 / INT_MIN, compacted with the same
    # extraction (T' = 1 selects exactly the masked entries, ascending)
    r0 = me * KR
    blk = kall_ref[pl.ds(r0, KR), :]
    pos = (r0 + lax.broadcasted_iota(I32, (KR, 128), 0)) * 128 + lax.broadcasted_iota(I32, (KR, 128), 1)
    sel = (blk > T) | ((blk == T) & (pos < P))
    mk_ref[...] = jnp.full((128, 128), jnp.int32(INT_MIN))
    if pay_ref is None:
        mk_ref[0:KR, :] = jnp.where(sel, jnp.int32(1), jnp.int32(INT_MIN))
        return extract_payload(mk_ref, 1, jnp.int32(1), jnp.int32(INT_MAX), K, own_v, None, pre_ref,
                               acc_ref, cc_ref, kc=kc)
    # pay_ref (KR, 128) i32 >= 0 per candidate slot: the selected slots' payloads compacted in slot
    # order into pay_out (INT_MIN past the count) by the same extraction (keys = payloads, T = 0)
    mk_ref[0:KR, :] = jnp.where(sel, pay_ref[...], jnp.int32(INT_MIN))
    return extract_payload(mk_ref, 1, jnp.int32(0), jnp.int32(INT_MAX), K, own_v, pay_out, pre_ref,
                           acc_ref, cc_ref, kc=kc)
