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
"""The MXU dot and the no-op tail behind TPU_GDN_PAIRED_HEADS, on their own.

Past a two-slot window's last real token, window_tile updates each state
with the decay selected to 1.0 at run time and k = +0 times v_new selected
to -0.0. Every state there is the previous position's update, a VPU
result, and the VPU must keep each such value bit for bit through that.
`test_noop_tail_keeps_every_vpu_result` checks it over every exponent class
and the special values. It runs first.

window_tile multiplies a [1, 128] FP32 row (k or q) with a value head's
[128, 128] FP32 state; with paired heads, with two heads' states side by
side, [128, 256]. Both forms round the state to BF16 for the MXU. Every
output lane of the paired dot must equal the matching single-head dot bit
for bit, on random operands and on operands that expose rounding,
accumulation order, signed zeros, subnormals and overflow. The paired dots
run as the window loop runs them: each right after another paired dot, of
a poison pair (both heads FP32 max, then both NaN), and a single-head dot
right after a paired dot. The single-head reference dots run in a kernel
of their own. Fast: run it before the kernel tests.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

D = 128
DIMS = (((2,), (1,)), ((0,), (0,)))
F32_MAX = np.finfo(np.float32).max
BF16_MAX = float(jnp.finfo(jnp.bfloat16).max)
TINY = np.finfo(np.float32).tiny  # smallest normal


def _tail_kernel(flag_ref, x_ref, a_ref, s_ref, kz_ref, out_ref):
    # window_tile's update past the last real token, decay * state +
    # k_t * v_new, with the decay and v_new selected from a flag read at run
    # time (flag 1: decay 1.0, v_new -0.0) and k = +0. out_ref[0]: y, the
    # multiply-add a * s + k * v_new as window_tile's update computes it,
    # out_ref[1]: that update past the last real token applied to y,
    # out_ref[2]: the same applied to the raw x.
    noop = jnp.full((1, D), flag_ref[0]) == 1
    decay = jnp.where(noop, 1.0, 2.0)
    v_new = jnp.where(noop, -0.0, 1.0)
    added = kz_ref[...] * v_new
    y = a_ref[...] * s_ref[...] + added
    out_ref[0] = y
    out_ref[1] = y * decay + added
    out_ref[2] = x_ref[...] * decay + added


def _f32_patterns(rng) -> np.ndarray:
    """FP32 bit patterns [rows, D]: both signs of every exponent with random
    mantissas (subnormals, NaN payloads), then +-0, +-inf, +-max, the
    smallest and largest subnormals, the smallest normals and quiet and
    signaling NaNs, padded with random patterns to whole registers."""
    exponents = np.arange(256, dtype=np.uint32) << 23
    mantissas = rng.integers(0, 1 << 23, (256, 64), dtype=np.uint32)
    bits = (exponents[:, None] | mantissas).reshape(-1)
    special = np.array(
        [
            0x00000000,
            0x7F800000,
            0x7F7FFFFF,
            0x00000001,
            0x007FFFFF,
            0x00800000,
            0x3F800000,
            0x7FC00000,
            0x7FFFFFFF,
            0x7F800001,
            0x7FBFFFFF,
        ],
        np.uint32,
    )
    bits = np.concatenate([bits, special])
    bits = np.concatenate([bits, bits | np.uint32(1 << 31)])
    pad = -len(bits) % (8 * D)
    bits = np.concatenate([bits, rng.integers(0, 1 << 32, pad, dtype=np.uint32)])
    return bits.reshape(-1, D)


def _f32_kind(bits: np.ndarray) -> np.ndarray:
    exponent, mantissa = (bits >> 23) & 0xFF, bits & 0x7FFFFF
    return np.select(
        [
            (exponent == 0) & (mantissa == 0),
            exponent == 0,
            (exponent == 255) & (mantissa == 0),
            (exponent == 255) & (mantissa >= 1 << 22),
            exponent == 255,
        ],
        ["zero", "subnormal", "inf", "quiet NaN", "signaling NaN"],
        "normal",
    )


def _vpu_operands(bits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(a, s) FP32 with a * s exactly the FP32 pattern `bits`: finite
    nonzero values as a normal value times a power of two (2^-64 or 2^64),
    so subnormals come out of a product of normal operands; zeros, infs and
    NaNs times 1.0."""
    exponent, mantissa = (bits >> 23) & 0xFF, bits & 0x7FFFFF
    finite = (exponent < 255) & ((exponent > 0) | (mantissa > 0))
    a, s = bits.view(np.float32).copy(), np.ones(bits.shape, np.float32)
    x = bits[finite].view(np.float32).astype(np.float64)
    shift = np.where(exponent[finite] < 191, 64.0, -64.0)
    a[finite] = (x * 2.0**shift).astype(np.float32)
    s[finite] = (2.0**-shift).astype(np.float32)
    assert np.array_equal(a[finite].astype(np.float64) * s[finite], x)
    return a, s


def test_noop_tail_keeps_every_vpu_result():
    bits = _f32_patterns(np.random.default_rng(0))
    x = bits.view(np.float32)
    a, s = _vpu_operands(bits)
    spec = pl.BlockSpec(x.shape, lambda: (0, 0))
    out = pl.pallas_call(
        _tail_kernel,
        in_specs=[pl.BlockSpec(memory_space=pltpu.SMEM)] + [spec] * 4,
        out_specs=pl.BlockSpec((3, *x.shape), lambda: (0, 0, 0)),
        out_shape=jax.ShapeDtypeStruct((3, *x.shape), jnp.float32),
    )(
        jnp.ones((1,), jnp.int32),
        jnp.asarray(x),
        jnp.asarray(a),
        jnp.asarray(s),
        jnp.zeros(x.shape, jnp.float32),
    )
    y, tail, raw = np.asarray(out).view(np.uint32)
    kinds = _f32_kind(bits)

    def per_kind(differ):
        return "; ".join(
            f"{kind}: {np.count_nonzero(differ & (kinds == kind))} of "
            f"{np.count_nonzero(kinds == kind)}"
            for kind in sorted(set(kinds[differ].tolist()))
        )

    # A record, not a requirement: the patterns the multiply-add does not
    # give back exactly (flushed or canonicalized), and the update applied to
    # raw patterns, which never reach it in window_tile.
    print(f"VPU result != pattern: {per_kind(y != bits) or 'none'}")
    print(f"update of the raw pattern changes it: {per_kind(raw != bits) or 'none'}")
    bad = per_kind(tail != y)
    assert not bad, "x * 1.0 + (+0) * -0.0 changes VPU results: " + bad


def _dot(row, rhs):
    # window_tile's head dot: [1, D] x [D, n] -> [1, n], FP32 accumulation.
    return jax.lax.dot(
        row[None], rhs[None], dimension_numbers=DIMS, preferred_element_type=jnp.float32
    )[0]


def _pair_kernel(row_ref, state_ref, poison_ref, out_ref):
    # Output rows: 0 / 1 the case's paired dot right after a paired dot of
    # poison pair 0 (FP32 max), 2 / 3 the same after poison pair 1 (NaN); 4
    # the single-head dots of head 0 and head 1, each right after a paired
    # dot; 8-11 the poison products. Mosaic issues independent dots to the
    # two MXUs in turn, so each poison and case dot runs twice in a row: one
    # poison dot right before a case dot on each MXU.
    row = row_ref[0, 0:1, :]
    for p in range(2):
        for m in range(2):
            out_ref[0, 8 + 2 * p + m : 9 + 2 * p + m, :] = _dot(row, poison_ref[p])
        for m in range(2):
            # Loaded again after each store, so the two dots stay two.
            pair = jnp.concat([state_ref[0, 0], state_ref[0, 1]], axis=1)
            out_ref[0, 2 * p + m : 2 * p + m + 1, :] = _dot(row, pair)
    for h in range(2):
        out_ref[0, 4:5, h * D : (h + 1) * D] = _dot(row, state_ref[0, h])


def _single_kernel(row_ref, state_ref, out_ref):
    row = row_ref[0, 0:1, :]
    out_ref[0, 0:1, :] = jnp.concat(
        [_dot(row, state_ref[0, 0]), _dot(row, state_ref[0, 1])], axis=1
    )


def _dots(rows: np.ndarray, states: np.ndarray):
    """rows [n, D], states [n, 2, D, D] FP32 -> {form: [n, 2 * D]}: the
    single-head dots from a kernel of their own, the paired dots after a
    poison pair (twice per pair) and the single-head dots after a paired
    dot."""
    n = rows.shape[0]
    rows8 = np.zeros((n, 8, D), np.float32)
    rows8[:, 0] = rows
    poison = np.stack(
        [np.full((D, 2 * D), F32_MAX, np.float32), np.full((D, 2 * D), np.nan)]
    ).astype(np.float32)
    row_spec = pl.BlockSpec((1, 8, D), lambda i: (i, 0, 0))
    state_spec = pl.BlockSpec((1, 2, D, D), lambda i: (i, 0, 0, 0))
    params = pltpu.CompilerParams(dimension_semantics=("arbitrary",))
    single = pl.pallas_call(
        _single_kernel,
        grid=(n,),
        in_specs=[row_spec, state_spec],
        out_specs=pl.BlockSpec((1, 8, 2 * D), lambda i: (i, 0, 0)),
        out_shape=jax.ShapeDtypeStruct((n, 8, 2 * D), jnp.float32),
        compiler_params=params,
    )(jnp.asarray(rows8), jnp.asarray(states))
    paired = pl.pallas_call(
        _pair_kernel,
        grid=(n,),
        in_specs=[
            row_spec,
            state_spec,
            pl.BlockSpec((2, D, 2 * D), lambda i: (0, 0, 0)),
        ],
        out_specs=pl.BlockSpec((1, 16, 2 * D), lambda i: (i, 0, 0)),
        out_shape=jax.ShapeDtypeStruct((n, 16, 2 * D), jnp.float32),
        compiler_params=params,
    )(jnp.asarray(rows8), jnp.asarray(states), jnp.asarray(poison))
    paired = np.asarray(paired)
    return {
        "single": np.asarray(single)[:, 0],
        "paired after a max pair": paired[:, 0],
        "paired after a max pair, again": paired[:, 1],
        "paired after a NaN pair": paired[:, 2],
        "paired after a NaN pair, again": paired[:, 3],
        "single after paired": paired[:, 4],
    }


def _bf16_exact(x):
    return np.asarray(
        jnp.asarray(x, jnp.float32).astype(jnp.bfloat16).astype(jnp.float32)
    )


def _cases(seed: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """name -> (row [D], states [2, D, D]); states are (column halves) the
    paired dot's first and second head."""
    rng = np.random.default_rng(seed)

    def normal(*shape, scale=1.0):
        return (scale * rng.standard_normal(shape)).astype(np.float32)

    def unit_row():
        r = normal(D)
        return (r / np.linalg.norm(r)).astype(np.float32)

    def log_uniform(shape, lo, hi):
        mag = np.exp(rng.uniform(np.log(lo), np.log(hi), shape))
        return (mag * rng.choice([-1.0, 1.0], shape)).astype(np.float32)

    cases = {}

    def add(name, row, s0, s1):
        cases[name] = (
            np.asarray(row, np.float32),
            np.stack([s0, s1]).astype(np.float32),
        )
        # The same pair the other way round, so each matrix also travels the
        # other half of the paired weight.
        cases[name + "/swapped"] = (
            np.asarray(row, np.float32),
            np.stack([s1, s0]).astype(np.float32),
        )

    # Served-like: unit-norm k / q rows, FP32 states (not BF16-exact).
    for i in range(4):
        add(f"random{i}", unit_row(), normal(D, D, scale=0.1), normal(D, D, scale=0.1))
    add(
        "bf16_exact",
        _bf16_exact(unit_row()),
        _bf16_exact(normal(D, D)),
        _bf16_exact(normal(D, D)),
    )
    # States whose FP32 values sit on, just below and just above BF16
    # rounding ties (round to nearest even), and rows with low mantissa bits.
    base = _bf16_exact(normal(D, D))
    ulp = np.abs(base) * 2.0**-8
    ties = (base + np.where(rng.random((D, D)) < 0.5, 0.5, -0.5) * ulp).astype(
        np.float32
    )
    near = (ties + rng.choice([-1, 1], (D, D)) * np.abs(ties) * 2.0**-20).astype(
        np.float32
    )
    row = unit_row()
    row_bits = (row.view(np.uint32) | np.uint32(0x7FFF)).view(np.float32)
    add("bf16_ties", row_bits, ties, near)
    # Wide dynamic range: sums depend on the order and grouping of additions.
    add(
        "dynamic_range",
        log_uniform(D, 1e-3, 1e3),
        log_uniform((D, D), 1e-20, 1e20),
        log_uniform((D, D), 1e-20, 1e20),
    )
    big = np.zeros((D, D), np.float32)
    for c in range(D):
        # One large pair that cancels and small terms in between, placed at
        # column-dependent rows (inside and across groups of 4 rows).
        i, j = c % D, (c * 7 + 1 + c % 3) % D
        if i == j:
            j = (j + 1) % D
        big[i, c], big[j, c] = 1e8, -1e8
        big[(i + 2) % D, c] = 1.0 + c * 2.0**-7
    add("cancel_order", np.ones(D, np.float32), big, -big)
    # Signed zeros: the zero k / q row of a position past the last token,
    # all-negative and all -0.0 columns, and products that cancel to zero.
    neg = -np.abs(normal(D, D))
    negz = np.full((D, D), -0.0, np.float32)
    mixed = np.where(rng.random((D, D)) < 0.5, neg, negz).astype(np.float32)
    add("zero_row", np.zeros(D, np.float32), neg, normal(D, D))
    add("neg_zero_row", np.full(D, -0.0, np.float32), neg, mixed)
    add("neg_zero_states", unit_row(), negz, mixed)
    add("pos_row_neg_columns", np.abs(unit_row()), neg, negz)
    pair = np.zeros((D, D), np.float32)
    pair[0::2], pair[1::2] = 0.375, -0.375
    add("exact_cancel", np.ones(D, np.float32), pair, -pair)
    # Subnormals in the states, in the row, and products that underflow.
    sub = (rng.integers(1, 1 << 23, (D, D)).astype(np.uint32)).view(np.float32)
    sub = np.where(rng.random((D, D)) < 0.5, sub, -sub).astype(np.float32)
    sub_bf16 = _bf16_exact(sub)
    add("subnormal_states", unit_row(), sub, sub_bf16)
    add(
        "subnormal_mixed",
        unit_row(),
        np.where(rng.random((D, D)) < 0.1, sub, normal(D, D)),
        normal(D, D),
    )
    sub_row = (rng.integers(1, 1 << 23, D).astype(np.uint32)).view(np.float32)
    add("subnormal_row", sub_row, normal(D, D), sub)
    add(
        "underflow",
        np.full(D, 1e-20, np.float32),
        np.full((D, D), 1e-20, np.float32),
        normal(D, D, scale=TINY),
    )
    # Large finite values: BF16 max, FP32 values that round to BF16 inf,
    # sums that overflow FP32, and +big / -big columns (inf - inf).
    add(
        "bf16_max",
        unit_row(),
        np.full((D, D), BF16_MAX, np.float32),
        np.full((D, D), -BF16_MAX, np.float32),
    )
    add("bf16_overflow", unit_row(), np.full((D, D), F32_MAX, np.float32), normal(D, D))
    add(
        "sum_overflow",
        np.ones(D, np.float32),
        np.full((D, D), 1e38, np.float32),
        np.full((D, D), -1e38, np.float32),
    )
    pm = np.where(np.arange(D)[:, None] % 2 == 0, BF16_MAX, -BF16_MAX) * np.ones((1, D))
    add("inf_minus_inf", np.ones(D, np.float32), pm.astype(np.float32), normal(D, D))
    # One head's state non-finite, the other finite (the finite head must not
    # see it), and one head all zero (zero data beside the other head).
    nan_state = normal(D, D)
    nan_state[rng.random((D, D)) < 0.05] = np.nan
    inf_state = normal(D, D)
    inf_state[rng.random((D, D)) < 0.05] = np.inf
    add("nan_head", unit_row(), nan_state, normal(D, D))
    add("inf_head", unit_row(), inf_state, normal(D, D))
    add("zero_head", unit_row(), np.zeros((D, D), np.float32), normal(D, D))
    return cases


@pytest.mark.parametrize("seed", [0, 1])
def test_paired_dot_matches_single_head_dots(seed):
    cases = _cases(seed)
    names = sorted(cases)
    rows = np.stack([cases[n][0] for n in names])
    states = np.stack([cases[n][1] for n in names])
    forms = _dots(rows, states)
    single = forms.pop("single")
    # The harness computes real products: served-like cases are close to an
    # FP64 reference of the BF16-rounded states.
    for i, name in enumerate(names):
        if name.startswith("random"):
            ref = rows[i].astype(np.float64) @ np.concatenate(
                [_bf16_exact(states[i, 0]), _bf16_exact(states[i, 1])], axis=1
            ).astype(np.float64)
            np.testing.assert_allclose(
                single[i], ref, rtol=2e-2, atol=2e-2, err_msg=name
            )
            assert np.count_nonzero(single[i]) > D, name
    bad = []
    for form, got in forms.items():
        differ = single.view(np.uint32) != got.view(np.uint32)
        bad += [
            f"{form} {name}: {np.count_nonzero(differ[i])} lanes"
            for i, name in enumerate(names)
            if differ[i].any()
        ]
    assert not bad, "differs from the single-head dots: " + "; ".join(bad)
