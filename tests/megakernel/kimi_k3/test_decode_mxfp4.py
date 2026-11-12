# SPDX-License-Identifier: Apache-2.0
"""The decode's masked MXFP4 products: running the scale-block products in
narrower contraction steps leaves every block partial, and so the gate/up
results, unchanged (Pallas interpret mode).

The activations are small integers, so every block's sum is exact and the
comparison does not depend on the order in which a backend accumulates a
product's contraction (the CPU's dot reorders it with the operand shapes).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl

from vllm_torchtpu.megakernel.kimi_k3.kernels import decode_megakernel as dm

pytestmark = pytest.mark.cpu_test

CONTRACTION, WIDTH = 1024, 256  # two pair tiles; gate/up of 2 x 128 columns


def _expert_weights(rng, slots):
    packed = rng.integers(
        0, 2**32, (slots, CONTRACTION // 8, 2 * WIDTH), dtype=np.uint64
    )
    scales = rng.integers(
        110, 124, (slots, CONTRACTION // 32, 2 * WIDTH), dtype=np.uint8
    )
    return packed.astype(np.uint32), scales


def _activation(rng, shape):
    return rng.integers(-8, 9, shape).astype(np.float32)


def test_scale_block_partials_do_not_depend_on_the_step():
    rng = np.random.default_rng(0)
    activation = jnp.asarray(_activation(rng, (4, 512)), jnp.bfloat16)
    values = jnp.asarray(rng.integers(-6, 7, (512, 384)) / 2, jnp.float8_e4m3fn)
    whole = dm._scale_block_partials(activation, values, 512)
    assert whole.shape == (4, 16, 384)
    for step in (128, 256):
        np.testing.assert_array_equal(
            np.asarray(dm._scale_block_partials(activation, values, step)),
            np.asarray(whole),
        )


def _block_tree(products):
    """The scale-block reduction's order on device, in float32: per row the
    8-block groups left to right, then blocks i + (i + 4), then the sums two
    apart, then the last pair."""
    rows, blocks, width = products.shape
    totals = np.empty((rows, width), np.float32)
    for row in range(rows):
        sums = products[row, 0:8]
        for group in range(8, blocks, 8):
            sums = sums + products[row, group : group + 8]
        fours = [sums[i] + sums[i + 4] for i in range(4)]
        totals[row] = (fours[0] + fours[2]) + (fours[1] + fours[3])
    return totals


@pytest.mark.parametrize("rows", [4, 8, 16])
@pytest.mark.parametrize("blocks", [8, 16, 24])
def test_scaled_block_sum_keeps_the_block_order(rows, blocks):
    """The rows' shared trees add exactly the reduction's pairs: with partials
    spread over many binary orders, any other order changes bits."""
    rng = np.random.default_rng(rows * blocks)
    spread = np.exp2(rng.integers(-24, 24, (rows, blocks, 256)))
    partial = (rng.standard_normal((rows, blocks, 256)) * spread).astype(np.float32)
    scale = np.exp2(rng.integers(-8, 8, (blocks, 256))).astype(np.float32)
    got = np.asarray(jax.jit(dm._scaled_block_sum)(partial, scale))
    want = _block_tree(partial * scale[None])
    np.testing.assert_array_equal(got.view(np.uint32), want.view(np.uint32))
    # These partials detect a different order: a sequential sum differs.
    products = partial * scale[None]
    sequential = products[:, 0]
    for block in range(1, blocks):
        sequential = sequential + products[:, block]
    assert not np.array_equal(sequential, want)


def _paired(monkeypatch, step, source, packed, scales, second=None, active=None):
    monkeypatch.setattr(dm, "MXFP4_PAIR_STEP", step)
    rows = source.shape[0] * (2 if second is not None else 1)

    def kernel(*refs):
        if second is None:
            source_ref, packed_ref, scales_ref, out_ref = refs
            second_ref = None
        else:
            source_ref, second_ref, packed_ref, scales_ref, out_ref = refs
        dm._paired_expert_gate_up(
            source_ref,
            packed_ref,
            scales_ref,
            out_ref,
            0,
            1,
            CONTRACTION,
            second_source=second_ref,
            active_capacity=active,
        )

    operands = (source,) if second is None else (source, second)
    return np.asarray(
        pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((rows, 4 * WIDTH), jnp.float32),
            interpret=True,
        )(*operands, packed, scales)
    )


def _single(monkeypatch, step, source, packed, scales):
    monkeypatch.setattr(dm, "MXFP4_PAIR_STEP", step)

    def kernel(source_ref, packed_ref, scales_ref, out_ref):
        dm._paired_order_expert_gate_up(
            source_ref, packed_ref, scales_ref, out_ref, CONTRACTION, 512
        )

    return np.asarray(
        pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((source.shape[0], 2 * WIDTH), jnp.float32),
            interpret=True,
        )(source, packed[0], scales[0])
    )


@pytest.mark.parametrize("rows", [1, 2, 4, 16])
def test_paired_gate_up_steps_are_bit_identical(monkeypatch, rows):
    rng = np.random.default_rng(rows)
    packed, scales = _expert_weights(rng, 2)
    source = _activation(rng, (rows, CONTRACTION))
    whole = _paired(monkeypatch, 512, source, packed, scales)
    stepped = _paired(monkeypatch, 256, source, packed, scales)
    assert np.count_nonzero(whole) > whole.size // 2
    np.testing.assert_array_equal(stepped.view(np.uint32), whole.view(np.uint32))
    # The single-expert projection keeps the paired arithmetic.
    single = _single(monkeypatch, 256, source, packed, scales)
    np.testing.assert_array_equal(
        single.view(np.uint32), whole[:, : 2 * WIDTH].view(np.uint32)
    )


@pytest.mark.parametrize("active", [None, 2, 8])
def test_compact_paired_gate_up_steps_are_bit_identical(monkeypatch, active):
    rng = np.random.default_rng(10 + (active or 0))
    packed, scales = _expert_weights(rng, 2)
    first, second = _activation(rng, (2, 8, CONTRACTION))
    whole = _paired(monkeypatch, 512, first, packed, scales, second, active)
    stepped = _paired(monkeypatch, 256, first, packed, scales, second, active)
    np.testing.assert_array_equal(stepped.view(np.uint32), whole.view(np.uint32))


@pytest.mark.parametrize(("rows", "active"), [(1, None), (8, 2), (8, 8)])
def test_shared_rows_project_both_halves(monkeypatch, rows, active):
    """EP16 halves share their rows, so a pair projects them once for both
    halves: the single-source product holds the two-source product's diagonal
    blocks (rows past the active capacity stay zero)."""
    rng = np.random.default_rng(3 + rows + (active or 0))
    packed, scales = _expert_weights(rng, 2)
    source = _activation(rng, (rows, CONTRACTION))
    single = _paired(monkeypatch, 256, source, packed, scales, active=active)
    pair = _paired(monkeypatch, 256, source, packed, scales, source, active)
    half = 2 * WIDTH
    np.testing.assert_array_equal(
        single[:, :half].view(np.uint32), pair[:rows, :half].view(np.uint32)
    )
    np.testing.assert_array_equal(
        single[:, half:].view(np.uint32), pair[rows:, half:].view(np.uint32)
    )
    assert not np.any(single[active or rows :])
