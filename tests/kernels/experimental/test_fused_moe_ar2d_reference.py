# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""The AR2D summation-order model against a rank-by-rank run of the exchange.

The simulation plays the eight ranks literally: who sends which rows and columns
to whom at s1, r1..r4 and s3, each rank summing what it holds in f32 and what it
received as bf16. CPU only, NumPy."""

import numpy as np
import pytest

from vllm_torchtpu.kernels.experimental.fused_moe_decode.ar2d_reference import (
    BF16,
    N_RANKS,
    all_to_all_reference,
    ar2d_mismatches,
    ar2d_reference,
    ar2d_reference_bf16_operands,
    assert_ar2d_bits,
)


def _wire(a):
    """What a copy carries: bf16, read back as f32."""
    return a.astype(BF16).astype(np.float32)


def _simulate(acc, chunks=1):
    """Outputs [8, R, h] of the protocol played rank by rank, rows cut in `chunks`."""
    x = np.asarray(acc, np.float32)
    _, rows, h = x.shape
    half, part = h // 2, h // 4
    out = np.zeros((N_RANKS, rows, h), BF16)
    rk = rows // chunks
    for k in range(chunks):
        rs = slice(k * rk, (k + 1) * rk)
        xs = x[:, rs]
        r2, r4 = rk // 2, rk // 4
        # (c, yb, xb) of each rank
        bits = {me: (me & 1, (me >> 1) & 1, me >> 2) for me in range(N_RANKS)}
        # s1: the mate sends bf16 of its partial of this rank's half.
        A = {}
        for me, (c, _, _) in bits.items():
            cols = slice(c * half, (c + 1) * half)
            A[me] = xs[me][:, cols] + _wire(xs[me ^ 1][:, cols])
        A0 = {me: A[me][:, :part] for me in A}
        A1 = {me: A[me][:, part:] for me in A}
        # r1: part 0 row halves across me ^ 4 (xb), part 1 across me ^ 2 (yb).
        B0, B1 = {}, {}
        for me, (_, yb, xb) in bits.items():
            hx, hy = slice(xb * r2, (xb + 1) * r2), slice(yb * r2, (yb + 1) * r2)
            B0[me] = A0[me][hx] + _wire(A0[me ^ 4][hx])
            B1[me] = A1[me][hy] + _wire(A1[me ^ 2][hy])
        # r2: B0 row quarters across me ^ 2, B1 across me ^ 4; rounded once.
        C0, C1 = {}, {}
        for me, (_, yb, xb) in bits.items():
            qy, qx = slice(yb * r4, (yb + 1) * r4), slice(xb * r4, (xb + 1) * r4)
            C0[me] = (B0[me][qy] + _wire(B0[me ^ 2][qy])).astype(BF16)
            C1[me] = (B1[me][qx] + _wire(B1[me ^ 4][qx])).astype(BF16)
        # r3: gather C0 across me ^ 2, C1 across me ^ 4, each into its owner's position.
        B0f, B1f = {}, {}
        for me, (_, yb, xb) in bits.items():
            b0 = np.zeros((r2, part), BF16)
            b0[yb * r4 : (yb + 1) * r4] = C0[me]
            b0[(1 - yb) * r4 : (2 - yb) * r4] = C0[me ^ 2]
            b1 = np.zeros((r2, part), BF16)
            b1[xb * r4 : (xb + 1) * r4] = C1[me]
            b1[(1 - xb) * r4 : (2 - xb) * r4] = C1[me ^ 4]
            B0f[me], B1f[me] = b0, b1
        # r4: gather B0f across me ^ 4, B1f across me ^ 2.
        AF = {}
        for me, (_, yb, xb) in bits.items():
            a0 = np.zeros((rk, part), BF16)
            a0[xb * r2 : (xb + 1) * r2] = B0f[me]
            a0[(1 - xb) * r2 : (2 - xb) * r2] = B0f[me ^ 4]
            a1 = np.zeros((rk, part), BF16)
            a1[yb * r2 : (yb + 1) * r2] = B1f[me]
            a1[(1 - yb) * r2 : (2 - yb) * r2] = B1f[me ^ 2]
            AF[me] = np.concatenate([a0, a1], axis=1)
        # s3: own half from AF, the other from the mate's AF.
        for me, (c, _, _) in bits.items():
            out[me, rs, c * half : (c + 1) * half] = AF[me]
            out[me, rs, (1 - c) * half : (2 - c) * half] = AF[me ^ 1]
    return out


def _acc(rows, h, seed):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(N_RANKS, rows, h)).astype(np.float32)


@pytest.mark.parametrize(
    "rows,h,chunks",
    [(4, 16, 1), (16, 64, 1), (16, 64, 2), (64, 2560, 1), (128, 2560, 2)],
)
def test_model_is_the_protocol(rows, h, chunks):
    acc = _acc(rows, h, seed=rows + h + chunks)
    sim = _simulate(acc, chunks)
    assert_ar2d_bits(sim, acc, chunks)


def test_cancelling_accumulators():
    """256, 1, -256 on ranks 0, 2, 4 (three chips), zeros elsewhere. Today and
    part 0 give 1 (256 - 256 cancels first); part 1 gives 1 on rows where the
    rank holding 257 keeps it (u = 0) and 0 where it is sent (u = 1, bf16(257)
    = 256). Revision 2 gave 0 on every part-1 row."""
    m, h = 64, 2560
    acc = np.zeros((N_RANKS, m, h), np.float32)
    acc[0], acc[2], acc[4] = 256.0, 1.0, -256.0
    ref = ar2d_reference(acc).astype(np.float32)
    u = (np.arange(m) % 32) // 16
    for c0 in (0, 1280):
        assert (ref[:, c0 : c0 + 640] == 1.0).all()
        part1 = ref[:, c0 + 640 : c0 + 1280]
        assert (part1 == np.where(u == 0, 1.0, 0.0)[:, None]).all()
    assert (all_to_all_reference(acc).astype(np.float32) == 1.0).all()
    rev2 = ar2d_reference_bf16_operands(acc).astype(np.float32)
    assert (rev2[:, 640:1280] == 0.0).all()


def test_close_to_the_f32_sum():
    acc = _acc(256, 2560, seed=3)
    exact = acc.astype(np.float64).sum(axis=0)
    ref = ar2d_reference(acc).astype(np.float64)
    assert np.abs(ref - exact).max() / np.abs(exact).max() < 1e-2


def test_local_operands_are_not_rounded():
    """The model reads the accumulators unrounded: rounding them first changes
    the result where an operand stays local."""
    acc = _acc(32, 2560, seed=11)
    a = ar2d_reference(acc).view(np.uint16)
    b = ar2d_reference(acc.astype(BF16).astype(np.float32)).view(np.uint16)
    assert (a != b).any()


def test_mismatch_report():
    acc = _acc(32, 2560, seed=5)
    good = np.broadcast_to(ar2d_reference(acc), (N_RANKS, 32, 2560)).copy()
    res = ar2d_mismatches(good, acc)
    assert res == dict(per_rank=[0] * N_RANKS, mismatched=0, ranks_identical=True)
    bad = good.copy()
    bad[3, 5, 700] = bad[3, 5, 700] + BF16(1.0)
    res = ar2d_mismatches(bad, acc)
    assert res["per_rank"][3] == 1 and res["mismatched"] == 1
    assert not res["ranks_identical"]
    with pytest.raises(AssertionError):
        assert_ar2d_bits(bad, acc)
    # The real rows of a padded call: outputs [8, T, h] against acc [8, m, h].
    assert ar2d_mismatches(good[:, :20], acc)["mismatched"] == 0
