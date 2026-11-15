# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""NumPy model of the summation order of the MoE decode kernel's AR2D exchange.

The exchange sums eight per-rank f32 accumulators of an [m, h] tensor. Ranks are
named by the mesh index the kernel uses for device ids, `me`, with peers `me ^ 1`
(chip mate), `me ^ 2` and `me ^ 4`; core c = me & 1 owns column half c. At every
sum the rank that makes it holds its own operand in f32 and the other operand
crossed a wire as bf16 (round to nearest even); every other rank receives a copy
of the result. The order for an element in column half c, rows cut per chunk:

1. Chip sums on core c: A(x, y) = acc[4x + 2y + c] + bf16(acc[4x + 2y + 1 - c]),
   in f32 (x = bit 2, y = bit 1 of the mesh index).
2. Part 0 of the half (its first h / 4 columns) sums across x, then y; part 1
   across y, then x. With t the row's half and u its quarter within that half
   (in the chunk), the rank whose first-axis bit is t keeps the row:
     part 0: B(y) = A(t, y) + bf16(A(1 - t, y)), f32
             C = bf16(B(u) + bf16(B(1 - u)))
     part 1: B(x) = A(x, t) + bf16(A(x, 1 - t)), f32
             C = bf16(B(u) + bf16(B(1 - u)))
3. Every rank ends with C.

`ar2d_reference_bf16_operands` is the earlier order (revision 2): every operand,
own or received, rounded to bf16 before each f32 sum; it depends on no row.
"""

from __future__ import annotations

import ml_dtypes
import numpy as np

N_RANKS = 8
BF16 = ml_dtypes.bfloat16


def round_bf16(a) -> np.ndarray:
    """`a` rounded to bf16 (round to nearest even), as the kernel's convert."""
    return np.asarray(a).astype(BF16)


def _rb(a: np.ndarray) -> np.ndarray:
    """f32 -> bf16 -> f32: an operand that crossed a wire."""
    return a.astype(BF16).astype(np.float32)


def ar2d_reference(acc, chunks: int = 1) -> np.ndarray:
    """The bits every rank must hold after the exchange.

    acc: [8, m, h] per-rank accumulators indexed by mesh index (taken as f32),
    all m rows of the call (padded rows included: they decide which rank keeps
    a row); m / chunks a multiple of 4, h a multiple of 4. Returns [m, h] bf16.
    """
    a32 = np.asarray(acc, np.float32)
    assert a32.ndim == 3 and a32.shape[0] == N_RANKS, a32.shape
    _, m, h = a32.shape
    assert h % 4 == 0 and m % chunks == 0 and (m // chunks) % 4 == 0, (m, h, chunks)
    half, part = h // 2, h // 4
    rk = m // chunks
    i = np.arange(m) % rk
    t = (i // (rk // 2))[:, None]
    u = ((i % (rk // 2)) // (rk // 4))[:, None]
    out = np.empty((m, h), BF16)
    for c in (0, 1):
        cols = slice(c * half, (c + 1) * half)
        A = {
            (x, y): a32[4 * x + 2 * y + c][:, cols]
            + _rb(a32[4 * x + 2 * y + 1 - c][:, cols])
            for x in (0, 1)
            for y in (0, 1)
        }
        # part 0: across x (kept by x = t), then y (kept by y = u)
        b0 = {
            y: np.where(t == 0, A[0, y] + _rb(A[1, y]), A[1, y] + _rb(A[0, y]))
            for y in (0, 1)
        }
        c0 = np.where(u == 0, b0[0] + _rb(b0[1]), b0[1] + _rb(b0[0]))
        # part 1: across y (kept by y = t), then x (kept by x = u)
        b1 = {
            x: np.where(t == 0, A[x, 0] + _rb(A[x, 1]), A[x, 1] + _rb(A[x, 0]))
            for x in (0, 1)
        }
        c1 = np.where(u == 0, b1[0] + _rb(b1[1]), b1[1] + _rb(b1[0]))
        out[:, c * half : c * half + part] = c0[:, :part].astype(BF16)
        out[:, c * half + part : (c + 1) * half] = c1[:, part:].astype(BF16)
    return out


def _add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a.astype(np.float32) + b.astype(np.float32)).astype(BF16)


def ar2d_reference_bf16_operands(partials) -> np.ndarray:
    """Revision 2's order: partials rounded to bf16, mate sums, then part 0
    across bit 2 then bit 1 and part 1 across bit 1 then bit 2, every pairwise
    sum f32 of two bf16 values rounded to bf16. [8, m, h] -> [m, h] bf16."""
    x = round_bf16(partials)
    assert x.ndim == 3 and x.shape[0] == N_RANKS, x.shape
    h = x.shape[2]
    assert h % 4 == 0, h
    p = {m: _add(x[m], x[m ^ 1]) for m in (0, 2, 4, 6)}
    bit2_first = _add(_add(p[0], p[4]), _add(p[2], p[6]))
    bit1_first = _add(_add(p[0], p[2]), _add(p[4], p[6]))
    out = np.empty_like(bit2_first)
    half, part = h // 2, h // 4
    for c0 in (0, half):
        out[:, c0 : c0 + part] = bit2_first[:, c0 : c0 + part]
        out[:, c0 + part : c0 + half] = bit1_first[:, c0 + part : c0 + half]
    return out


def all_to_all_reference(acc) -> np.ndarray:
    """Today's exchange: partials rounded to bf16, summed in f32 in rank order,
    rounded once. [8, m, h] -> [m, h] bf16."""
    x = round_bf16(acc).astype(np.float32)
    s = x[0]
    for r in range(1, N_RANKS):
        s = s + x[r]
    return s.astype(BF16)


def ar2d_mismatches(outputs, acc, chunks: int = 1) -> dict:
    """Compares the eight ranks' outputs [8, T, h] (T <= m: the call's real
    rows) with the model of acc [8, m, h], bit for bit.

    Returns the mismatched element count per rank, their total, and whether all
    ranks hold identical bits.
    """
    out = round_bf16(outputs).view(np.uint16)
    ref = ar2d_reference(acc, chunks)[: out.shape[1]].view(np.uint16)
    assert out.shape == (N_RANKS,) + ref.shape, (out.shape, ref.shape)
    per_rank = [int((out[r] != ref).sum()) for r in range(N_RANKS)]
    return dict(
        per_rank=per_rank,
        mismatched=sum(per_rank),
        ranks_identical=bool((out == out[:1]).all()),
    )


def assert_ar2d_bits(outputs, acc, chunks: int = 1) -> None:
    """Asserts every rank's output equals the model in bits."""
    res = ar2d_mismatches(outputs, acc, chunks)
    assert res["mismatched"] == 0 and res["ranks_identical"], res
