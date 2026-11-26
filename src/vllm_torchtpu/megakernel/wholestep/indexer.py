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
"""QSA decode indexer on the TensorCore: score, exact top-K, compaction (no scalar
loop).

Stock semantics (`kernels/qwen4_exp`, `streamindex_topk` decode path):

  iq      = index_rope(gemma_norm(mixed @ W_idx[:, :512] per head, idx_q_ln))   [rows,
  4, 128]
  score_g = sum_h relu(iq_h . ck_g) * 128^-0.5     over complete groups g < n_vis = len
  // 4
  sel     = top-K of score (ties -> lower group id); n_vis <= K selects every visible
  group

Selection (measured exact on every row, ~2 µs/row, P1b j-988c4e1c): bisection on the f32
bit pattern with per-row counts as a vector, a tie-break bisection on ids, then
compaction
by matmuls: in-row ranks `P = sel @ U`, row offsets `O = c @ Us`, the row of output slot
j by
a compare-and-count, that row's ranks by a one-hot matmul, the lane by another count.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ..ops import BF16, F32, I32, rmsnorm, rnd

IDX_HEADS, IDX_DIM = 4, 128
I16 = jnp.int16


def _next_toward(x: jax.Array, up: jax.Array, bitcast: Any) -> jax.Array:
    """`nextafter(x, +inf if up else -inf)` for finite nonzero f32 `x`, by bit
    arithmetic."""
    b = bitcast(x, I32)
    pos = (
        x > 0
    )  # magnitude grows toward +inf for x > 0 (logic ops: Mosaic rejects bool selects)
    inc = (pos & up) | (~pos & ~up)
    return bitcast(jnp.where(inc, b + 1, b - 1), F32)


def fma_bf16(a: jax.Array, b: jax.Array, c: jax.Array, bitcast: Any) -> jax.Array:
    """Stock `_fma`: BF16 of the exact `a * b + c` (two-sum residual breaks BF16
    ties).
    """
    product = a * b
    high = product + c
    z = high - product
    residual = (product - (high - z)) + (c - z)
    tie = (bitcast(high, I32) & 0xFFFF) == 0x8000
    adj = jnp.where(
        tie & (residual != 0), _next_toward(high, residual > 0, bitcast), high
    )
    return rnd(adj)


def index_rope(
    x: jax.Array, cos: jax.Array, sin: jax.Array, rot: int, bitcast: Any
) -> jax.Array:
    """Stock `index_rope`: `[B(a c - B(b s)), B(a s + B(b c)), x[rot:]]` on rows
    of `x`.
    """
    half = rot // 2
    a, b = x[:, :half], x[:, half:rot]
    y1 = fma_bf16(a, cos, -rnd(b * sin), bitcast)
    y2 = fma_bf16(a, sin, rnd(b * cos), bitcast)
    return jnp.concatenate((y1, y2, x[:, rot:]), axis=-1)


def index_query(
    y: jax.Array, iqn: jax.Array, cos, sin, rot: int, eps: float, bitcast
) -> list:
    """Per head `[rows, 128]` BF16-valued queries from the projection `y [rows,
    >=512]`.
    """
    out = []
    for h in range(IDX_HEADS):
        q = rmsnorm(y[:, h * IDX_DIM : (h + 1) * IDX_DIM], iqn, eps, zero_centered=True)
        out.append(index_rope(q, cos, sin, rot, bitcast))
    return out


def score_rows(qr: jax.Array, ck: jax.Array) -> jax.Array:
    """`[T, G]` scores of `T` rows: `qr [T*4, 128]` (row-major over heads), `ck [G,
    128]`."""
    st = jax.lax.dot_general(
        qr, ck, (((1,), (1,)), ((), ())), preferred_element_type=F32
    )
    T = qr.shape[0] // IDX_HEADS
    return (jnp.maximum(st, 0.0) * IDX_DIM**-0.5).reshape(T, IDX_HEADS, -1).sum(axis=1)


def thresholds(bits: jax.Array, ids: jax.Array, nv: jax.Array, K: int, G: int,
               tie_skip: bool = False, early_exit: tuple = (), packed16: bool = False):
    """Per-row `(thr, tmax)` [R, 1, 1] for bit patterns `bits [R, S, 128]` (-1 =
    invisible).

    `tie_skip` (VLLM_TPU_MK_QSA_THR_FAST): the id bisection runs only when some row's tie
    at `thr` is truncated (`tmax` reads `t` only then). `early_exit = (bit, every)`: the
    value passes from `bit` down run in a `while_loop` (`every` passes per trip) that stops
    once every row over `K` has `cnt(bits >= lo) == K`: `{bits >= lo}` is then exactly the
    top-K set, no tie is truncated (`tmax = G`), and the compaction's `b >= lo` selects the
    same groups in the same order as `b >= thr` (a stopped `lo` may sit below the K-th
    value; nothing lies in between). Both: same selection as the plain passes.

    `packed16` (VLLM_TPU_MK_QSA_LANE_MAJOR_SELECT; needs `S % 16 == 0` and `G <= 65536`):
    the count passes compare int16 keys (`_value_passes16`, the id passes on `ids - 32768`),
    two groups per 32-bit lane: same counts, same results (same early-exit trips)."""
    cnt = lambda m: jnp.sum(m.astype(I32), axis=(1, 2), keepdims=True)  # noqa: E731
    over = nv > K
    lo = jnp.zeros(nv.shape, I32)
    packed = packed16 and bits.shape[1] % 16 == 0 and G <= 1 << 16
    if early_exit:
        if packed and _packed_early_ok(*early_exit):
            lo = _value_passes16_early(bits, over, K, *early_exit)
        else:
            lo = _value_passes_early(bits, cnt, over, K, *early_exit)
    elif packed:
        lo = _value_passes16(bits, K)
    else:
        for b in range(30, -1, -1):
            cand = lo | (1 << b)
            lo = jnp.where(cnt(bits >= cand) >= K, cand, lo)
    thr = jnp.where(over, lo, 0)
    eq = bits == thr
    need = K - cnt(bits > thr)

    def ties() -> jax.Array:
        t = jnp.zeros(nv.shape, I32)
        if packed:  # ids - 32768 where tied at thr, else 32767 (below no candidate)
            key = jnp.where(eq, (ids - 32768).astype(I16), I16(32767))
        for b in range(int(G - 1).bit_length() - 1, -1, -1):
            cand = t | (1 << b)
            if packed:
                n = _count16(key < (cand - 32768).astype(I16))
            else:
                n = cnt(eq & (ids < cand))
            t = jnp.where(n < need, cand, t)
        return t

    if not (tie_skip or early_exit):
        t = ties()
        return thr, jnp.where(over & (cnt(eq) > need), t, G)
    trunc = over & (cnt(eq) > need)
    if tie_skip:
        t = jax.lax.cond(
            jnp.max(trunc.astype(I32)) > 0, ties, lambda: jnp.zeros(nv.shape, I32)
        )
    else:
        t = ties()
    return thr, jnp.where(trunc, t, G)


def _count16(m: jax.Array) -> jax.Array:
    """Per-row counts `[R, 1, 1]` int32 of a mask `[R, S, 128]` of int16 compares: int16 sums
    of whole (16, 128) tiles (at most S / 16 per lane; Mosaic has no int16 reduction), then
    int32."""
    R, S, L = m.shape
    p = jnp.where(m, I16(1), I16(0)).reshape(R, S // 16, 16, L)
    parts = [p[:, i] for i in range(S // 16)]
    while len(parts) > 1:
        odd = parts[len(parts) - len(parts) % 2 :]
        parts = [a + b for a, b in zip(parts[::2], parts[1::2])] + odd
    return jnp.sum(parts[0].astype(I32), axis=(1, 2), keepdims=True)


def _value_passes16(bits: jax.Array, K: int) -> jax.Array:
    """`thresholds`' value bisection (`lo`) on int16 halves of `bits` (-1 or >= 0). Bits 30..16:
    a candidate's low half is 0, so `bits >= cand` iff `bits >> 16 >= cand >> 16`. Bits 15..0,
    with the high half `p` settled: `bits >= cand` iff `hi > p`, or `hi == p` and the low half
    is `>= cand & 0xFFFF` (compared biased by -32768; `-32768` for `hi != p` is below every
    candidate, whose low half is >= 1)."""
    hi = (bits >> 16).astype(I16)
    lo = jnp.zeros((bits.shape[0], 1, 1), I32)
    for b in range(30, 15, -1):
        cand = lo | (1 << b)
        lo = jnp.where(_count16(hi >= (cand >> 16).astype(I16)) >= K, cand, lo)
    p = (lo >> 16).astype(I16)
    low = jnp.where(hi == p, ((bits & 0xFFFF) - 32768).astype(I16), I16(-32768))
    k = K - _count16(hi > p)
    for b in range(15, -1, -1):
        cand = lo | (1 << b)
        lo = jnp.where(_count16(low >= ((cand & 0xFFFF) - 32768).astype(I16)) >= k, cand, lo)
    return lo


def _packed_early_ok(bit: int, every: int) -> bool:
    """`_value_passes16_early` keeps the int32 early exit's trips: the static passes stay in
    the high half (`bit >= 15`) and no trip straddles bit 16 (`every` divides 16)."""
    return bit >= 15 and 16 % every == 0 and (bit + 1) % every == 0


def _value_passes16_early(bits: jax.Array, over, K: int, bit: int, every: int) -> jax.Array:
    """`_value_passes_early` on `_value_passes16`'s int16 keys: the same exact counts per
    candidate, so the same `lo` updates and the same stopping trip. One `while_loop` for the
    high-half bits `[bit, 16]`, then the low-half keys against the settled high half `p`, and
    one for bits `[15, 0]` (counts `count(hi > p) + count(low >= cand's low half)`)."""
    hi = (bits >> 16).astype(I16)
    lo = jnp.zeros(over.shape, I32)
    n_ge = jnp.full(over.shape, -1, I32)  # unknown count: never "done"
    for b in range(30, bit, -1):
        cand = lo | (1 << b)
        n = _count16(hi >= (cand >> 16).astype(I16))
        lo, n_ge = jnp.where(n >= K, cand, lo), jnp.where(n >= K, n, n_ge)

    def pending(n_ge: Any) -> Any:
        return jnp.max((over & jnp.logical_not(_exact_top(n_ge, K))).astype(I32)) > 0

    def body_hi(c: Any) -> Any:
        b, lo, n_ge = c
        for i in range(every):
            cand = lo | jnp.left_shift(jnp.int32(1), b - i)
            n = _count16(hi >= (cand >> 16).astype(I16))
            lo, n_ge = jnp.where(n >= K, cand, lo), jnp.where(n >= K, n, n_ge)
        return b - every, lo, n_ge

    c = jax.lax.while_loop(
        lambda c: (c[0] >= 16) & pending(c[2]), body_hi, (jnp.int32(bit), lo, n_ge)
    )
    p = (c[1] >> 16).astype(I16)
    low = jnp.where(hi == p, ((bits & 0xFFFF) - 32768).astype(I16), I16(-32768))
    above = _count16(hi > p)

    def body_lo(c: Any) -> Any:
        b, lo, n_ge = c
        for i in range(every):
            cand = lo | jnp.left_shift(jnp.int32(1), b - i)
            n = above + _count16(low >= ((cand & 0xFFFF) - 32768).astype(I16))
            lo, n_ge = jnp.where(n >= K, cand, lo), jnp.where(n >= K, n, n_ge)
        return b - every, lo, n_ge

    return jax.lax.while_loop(lambda c: (c[0] >= 0) & pending(c[2]), body_lo, c)[1]


def _exact_top(n_ge: jax.Array, K: int) -> jax.Array:
    """Early-exit test of a row: its `{bits >= lo}` holds exactly `K` groups."""
    return n_ge == K


def _value_passes_early(bits, cnt, over, K: int, bit: int, every: int) -> jax.Array:
    """The value bisection's `lo`: static passes above `bit`, then a `while_loop` of
    `every` passes per trip from `bit` down that stops when no row over `K` is pending
    (`thresholds`). A row that is done and keeps going only moves `lo` up toward its K-th
    value (same top-K set), so the stop granularity does not change the selection."""
    assert 0 <= bit <= 30 and every >= 1 and (bit + 1) % every == 0, (bit, every)
    lo = jnp.zeros(over.shape, I32)
    n_ge = jnp.full(over.shape, -1, I32)  # unknown count: never "done"
    for b in range(30, bit, -1):
        cand = lo | (1 << b)
        n = cnt(bits >= cand)
        lo, n_ge = jnp.where(n >= K, cand, lo), jnp.where(n >= K, n, n_ge)

    def cond(c: Any) -> Any:
        b, _, n_ge = c
        pending = over & jnp.logical_not(_exact_top(n_ge, K))
        return (b >= 0) & (jnp.max(pending.astype(I32)) > 0)

    def body(c: Any) -> Any:
        b, lo, n_ge = c
        for i in range(every):
            cand = lo | jnp.left_shift(jnp.int32(1), b - i)
            n = cnt(bits >= cand)
            lo, n_ge = jnp.where(n >= K, cand, lo), jnp.where(n >= K, n, n_ge)
        return b - every, lo, n_ge

    return jax.lax.while_loop(cond, body, (jnp.int32(bit), lo, n_ge))[1]


def compact_row(b, th, tm, nv, K: int, consts: tuple, lane_major: bool = False) -> jax.Array:
    """Selected group ids of one row as a lane-major `[1, K]` int32 (slots past the
    count: 0).

    `b [S, 128]` bit patterns (-1 = invisible), `th / tm / nv` `[1, 1]`. `lane_major`
    (VLLM_TPU_MK_QSA_LANE_MAJOR_SELECT; `consts` from `compact_consts(..., lane_major=True)`):
    `compact_sel`'s lane-major compaction, same ids.
    """
    lid = consts[-1]
    if lane_major:
        ov = nv > K
        sel = (ov & ((b > th) | ((b == th) & (lid <= tm)))) | (~ov & (b >= 0))
        return compact_sel(sel, nv, K, consts)
    U, Us, jcol, srow, lid = consts
    S = b.shape[0]
    ov = nv > K
    sel = (ov & ((b > th) | ((b == th) & (lid <= tm)))) | (~ov & (b >= 0))
    P = jnp.dot(sel.astype(F32).astype(BF16), U, preferred_element_type=F32)
    c_row = jnp.transpose(jnp.broadcast_to(P[:, 127:128], (S, 128)))[0:1, :]
    O_row = exclusive_prefix(c_row, Us)
    E_row = O_row + c_row
    jf = jcol.astype(F32)
    r_j = jnp.sum((E_row <= jf).astype(I32), axis=1, keepdims=True)
    oh = srow == r_j
    Pg = jnp.dot(
        oh.astype(F32).astype(BF16), P.astype(BF16), preferred_element_type=F32
    )
    O_rj = jnp.sum(jnp.where(oh, O_row, 0.0), axis=1, keepdims=True)
    lane_j = jnp.sum((Pg <= jf - O_rj).astype(I32), axis=1, keepdims=True)
    # Slots past the row's count hold group 0, so DMAs issued for them stay in bounds.
    idc = jnp.where(jcol < jnp.minimum(nv, K), r_j * 128 + lane_j, 0).astype(F32)
    return jnp.transpose(jnp.broadcast_to(idc, (K, 128)))[0:1, :].astype(I32)


def compact_sel(sel: jax.Array, nv: jax.Array, K: int, consts: tuple) -> jax.Array:
    """`compact_row` of the selection mask `sel [S, 128]` (holding `nv` groups if `nv <= K`,
    else `K`), lane-major over the slots (`consts`: `compact_consts(..., lane_major=True)`).

    Slot `j` comes from sublane row `r_j` (rows ending at or before `j`) and lane `lane_j`
    (lanes of row `r_j` whose in-row rank is at or below `j - O[r_j]`). The in-row ranks come
    transposed `PT [128, S]` from the MXU, `r_j` and `lane_j` are sums over sublanes, and
    `P[r_j, :]` with `O[r_j]` is gathered by one one-hot matmul (no lane reductions, no final
    transpose). Integer-valued f32 throughout (counts <= 128 and offsets split into bytes are
    exact in BF16): the same ids as the sublane-major compaction."""
    Ut, Us, jrow, scol, _ = consts
    S = sel.shape[0]
    selb = sel.astype(F32).astype(BF16)
    PT = jax.lax.dot_general(  # PT[l, r]: selected lanes <= l of row r
        Ut, selb, (((1,), (1,)), ((), ())), preferred_element_type=F32
    )
    c_row = PT[127:128, :]
    O_row = exclusive_prefix(c_row, Us)
    E_col = jnp.transpose(jnp.broadcast_to(O_row + c_row, (128, S)))  # [S, 128]: E[r]
    jf = jrow.astype(F32)
    E = jnp.concatenate([E_col] * -(-K // 128), axis=1)[:, :K]
    r_j = sum_rows((E <= jf).astype(I32))
    oh = (scol == r_j).astype(F32).astype(BF16)  # [S, K]
    O_hi = jnp.floor(O_row * (1.0 / 256.0))
    lhs = jnp.concatenate([PT, O_hi, O_row - 256.0 * O_hi, jnp.zeros((6, S), F32)], axis=0)
    g = jnp.dot(lhs.astype(BF16), oh, preferred_element_type=F32)  # [136, K]
    O_rj = 256.0 * g[128:129] + g[129:130]
    lane_j = sum_rows((g[:128] <= jf - O_rj).astype(I32))
    # Slots past the row's count hold group 0, so DMAs issued for them stay in bounds.
    return jnp.where(jrow < jnp.minimum(nv, K), r_j * 128 + lane_j, 0)


def sum_rows(x: jax.Array) -> jax.Array:
    """`[1, W]` sum over the rows of `x [N, W]`: adds of whole 8-row tiles, then 8 rows."""
    n = x.shape[0]
    if n % 8 == 0 and n > 8:
        parts = [x[i : i + 8] for i in range(0, n, 8)]
        while len(parts) > 1:
            odd = parts[len(parts) - len(parts) % 2 :]
            parts = [a + c for a, c in zip(parts[::2], parts[1::2])] + odd
        x = parts[0]
    return jnp.sum(x, axis=0, keepdims=True)


def exclusive_prefix(c_row: jax.Array, us: jax.Array) -> jax.Array:
    """Exclusive prefix sum `[1, S]` of the per-sublane-row counts `c_row [1, S]` (each
    <= 128,
    exact in BF16): one `[1, 128] x [128, 128]` strict-triangular matmul per 128-lane
    chunk plus
    the running total of the chunks before it (f32 integers: exact). Replaces the `[S,
    S]` strict
    triangle, whose VMEM grows with S^2 (0.5 MiB at S = 512, 262K contexts). An S that
    is not a
    multiple of 128 (small test / 32K-cap shapes) keeps the one `[S, S]` matmul."""
    S = c_row.shape[1]
    if S % 128:
        si = jax.lax.broadcasted_iota(I32, (S, S), 0)
        sj = jax.lax.broadcasted_iota(I32, (S, S), 1)
        return jnp.dot(
            c_row.astype(BF16), (si < sj).astype(BF16), preferred_element_type=F32
        )
    out, carry = [], jnp.zeros((1, 1), F32)
    for c in range(c_row.shape[1] // 128):
        x = c_row[:, c * 128 : (c + 1) * 128]
        out.append(jnp.dot(x.astype(BF16), us, preferred_element_type=F32) + carry)
        carry = carry + jnp.sum(x, axis=1, keepdims=True)
    return out[0] if len(out) == 1 else jnp.concatenate(out, axis=1)


def compact_consts(S: int, K: int, lane_major: bool = False) -> tuple:
    li = jax.lax.broadcasted_iota(I32, (128, 128), 0)
    lj = jax.lax.broadcasted_iota(I32, (128, 128), 1)
    if lane_major:  # `compact_sel`
        return (
            (li >= lj).astype(BF16),  # transposed in-row prefix: PT = Ut . sel^T
            (li < lj).astype(BF16),  # 128-lane strict triangle of `exclusive_prefix`
            jax.lax.broadcasted_iota(I32, (1, K), 1),
            jax.lax.broadcasted_iota(I32, (S, 1), 0),
            jax.lax.broadcasted_iota(I32, (S, 128), 0) * 128
            + jax.lax.broadcasted_iota(I32, (S, 128), 1),
        )
    return (
        (li <= lj).astype(BF16),
        (li < lj).astype(BF16),  # 128-lane strict triangle of `exclusive_prefix`
        jax.lax.broadcasted_iota(I32, (K, 1), 0),
        jax.lax.broadcasted_iota(I32, (1, S), 1),
        jax.lax.broadcasted_iota(I32, (S, 128), 0) * 128
        + jax.lax.broadcasted_iota(I32, (S, 128), 1),
    )


# ------------------------------------------- visible-chunk selection (`select_nvis`)
# The functions below are `thresholds` / `compact_row` restricted to the sublane rows that
# hold visible groups: every other row is all -1 (invisible) and adds nothing to any count,
# so the same thresholds, ids and order come out of fewer passes over fewer rows.
def thresholds_chunks(load_bits, load_ids, n_ch: Any, nv: jax.Array, K: int, G: int,
                      shape: tuple, digit: int = 1):
    """`thresholds` over `n_ch` (traced) score chunks: `load_bits(c)` / `load_ids(c)` give
    chunk `c`'s bit patterns (-1 = invisible) / group ids, both `shape = [R, CH, 128]`; the
    rows outside the chunks must be invisible. Integer counts: any summation order.
    `digit` > 1: each pass settles `digit` bits of the bisection at once (the counts of all
    `2^digit - 1` candidates of the next digit; the count is monotone in the candidate, so
    the digit is the number of candidates that pass): the same values in fewer dependent
    passes."""
    if digit > 1:
        return _thresholds_digits(load_bits, load_ids, n_ch, nv, K, G, shape, digit)

    def cnt(*fs):
        def body(c: Any, accs: Any) -> Any:
            return tuple(a + f(c).astype(I32) for a, f in zip(accs, fs))

        accs = jax.lax.fori_loop(0, n_ch, body, tuple(jnp.zeros(shape, I32) for _ in fs))
        return [jnp.sum(a, axis=(1, 2), keepdims=True) for a in accs]

    over = nv > K
    lo = jnp.zeros(nv.shape, I32)
    for b in range(30, -1, -1):
        cand = lo | (1 << b)
        (n,) = cnt(lambda c, cand=cand: load_bits(c) >= cand)
        lo = jnp.where(n >= K, cand, lo)
    thr = jnp.where(over, lo, 0)
    n_gt, n_eq = cnt(lambda c: load_bits(c) > thr, lambda c: load_bits(c) == thr)
    need = K - n_gt
    t = jnp.zeros(nv.shape, I32)
    for b in range(int(G - 1).bit_length() - 1, -1, -1):
        cand = t | (1 << b)
        (n,) = cnt(lambda c, cand=cand: (load_bits(c) == thr) & (load_ids(c) < cand))
        t = jnp.where(n < need, cand, t)
    tmax = jnp.where(over & (n_eq > need), t, G)
    return thr, tmax


def _thresholds_digits(load_bits, load_ids, n_ch, nv, K, G, shape, digit):
    R, CH = shape[0], shape[1]
    part = (R, CH // 8, 8, 128) if CH % 8 == 0 else None

    def cnt(fs):
        """Counts `[R, 1, 1]` of the predicates `fs` (one pass over the chunks)."""
        acc_shape = (R, 8, 128) if part else shape

        def body(c: Any, accs: Any) -> Any:
            out = []
            for a, f in zip(accs, fs):
                m = f(c).astype(I32)
                out.append(a + (jnp.sum(m.reshape(part), axis=1) if part else m))
            return tuple(out)

        accs = jax.lax.fori_loop(0, n_ch, body, tuple(jnp.zeros(acc_shape, I32) for _ in fs))
        return [jnp.sum(a, axis=(1, 2), keepdims=True) for a in accs]

    def steps(top: int):
        hi = top
        while hi >= 0:
            lo_b = max(hi - digit + 1, 0)
            yield lo_b, hi - lo_b + 1
            hi = lo_b - 1

    over = nv > K
    lo = jnp.zeros(nv.shape, I32)
    for s, w in steps(30):
        cands = [lo | (k << s) for k in range(1, 1 << w)]
        ns = cnt([lambda c, cd=cd: load_bits(c) >= cd for cd in cands])
        k = sum((n >= K).astype(I32) for n in ns)
        lo = lo | (k << s)
    thr = jnp.where(over, lo, 0)
    n_gt, n_eq = cnt([lambda c: load_bits(c) > thr, lambda c: load_bits(c) == thr])
    need = K - n_gt
    t = jnp.zeros(nv.shape, I32)
    for s, w in steps(int(G - 1).bit_length() - 1):
        cands = [t | (k << s) for k in range(1, 1 << w)]
        ns = cnt([
            lambda c, cd=cd: (load_bits(c) == thr) & (load_ids(c) < cd) for cd in cands
        ])
        k = sum((n < need).astype(I32) for n in ns)
        t = t | (k << s)
    tmax = jnp.where(over & (n_eq > need), t, G)
    return thr, tmax


def chunk_consts(CR: int, K: int) -> tuple:
    """Constants of `compact_chunk` for `CR`-row chunks: `U`, the in-chunk strict triangle
    (`exclusive_prefix`'s 128-lane one at CR = 128), slot column, chunk row iota."""
    li = jax.lax.broadcasted_iota(I32, (128, 128), 0)
    lj = jax.lax.broadcasted_iota(I32, (128, 128), 1)
    si = jax.lax.broadcasted_iota(I32, (CR, CR), 0)
    sj = jax.lax.broadcasted_iota(I32, (CR, CR), 1)
    return (
        (li <= lj).astype(BF16),
        (si < sj).astype(BF16),
        jax.lax.broadcasted_iota(I32, (K, 1), 0),
        jax.lax.broadcasted_iota(I32, (1, CR), 1),
    )


def compact_chunk(sel: jax.Array, tot: jax.Array, row_of, consts: tuple):
    """One chunk of `compact_row`'s compaction (select-all of `sel`): the selected lanes of
    the chunk's rows `sel [CR, 128]` take output slots `[tot, tot + count)` (`tot [1, 1]` f32:
    the count of the chunks before it, which hold lower ids). Returns the lane-major `[1, K]`
    f32 slot ids of those slots (0 elsewhere; chunks add up) and the new `tot`. `row_of(r)`:
    the sublane row (id // 128) of the chunk's row `r [K, 1]`. Same integer arithmetic as
    `compact_row` (BF16-exact counts <= 128, f32-exact offsets)."""
    U, tri, jcol, srow = consts
    CR, K = sel.shape[0], jcol.shape[0]
    P = jnp.dot(sel.astype(F32).astype(BF16), U, preferred_element_type=F32)
    c_row = jnp.transpose(jnp.broadcast_to(P[:, 127:128], (CR, 128)))[0:1, :]
    O_row = jnp.dot(c_row.astype(BF16), tri, preferred_element_type=F32) + tot
    E_row = O_row + c_row
    jf = jcol.astype(F32)
    r_j = jnp.sum((E_row <= jf).astype(I32), axis=1, keepdims=True)
    oh = srow == r_j
    Pg = jnp.dot(
        oh.astype(F32).astype(BF16), P.astype(BF16), preferred_element_type=F32
    )
    O_rj = jnp.sum(jnp.where(oh, O_row, 0.0), axis=1, keepdims=True)
    lane_j = jnp.sum((Pg <= jf - O_rj).astype(I32), axis=1, keepdims=True)
    n = jnp.sum(c_row, axis=1, keepdims=True)
    live = (jf >= tot) & (jf < tot + n)
    idc = jnp.where(live, row_of(r_j) * 128 + lane_j, 0).astype(F32)
    return jnp.transpose(jnp.broadcast_to(idc, (K, 128)))[0:1, :], tot + n


# ---------------------------------------------------------------- reference
def topk_reference(scores: np.ndarray, nvis: int, K: int) -> list[int]:
    """Stock selection: top-K by score, ties to the lower id, over ids < nvis."""
    s = np.asarray(scores[:nvis], np.float32)
    return np.lexsort((np.arange(nvis), -s))[:K].tolist()
