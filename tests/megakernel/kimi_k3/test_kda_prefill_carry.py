# SPDX-License-Identifier: Apache-2.0
"""Conv history carried by the tiled KDA prefill kernel, plain and packed."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.megakernel.kimi_k3.kernels import prefill_megakernel as kernels

pytestmark = pytest.mark.cpu_test

TOKENS, HIDDEN, TILE = 384, 128, 128


def kda_inputs(seed, lead=()):
    rng = np.random.default_rng(seed)

    def normal(*shape, dtype=jnp.float32):
        return jnp.asarray(rng.normal(size=shape), dtype)

    # Integer operands make the projection exact in any summation order.
    def integers(*shape):
        return jnp.asarray(rng.integers(-2, 3, size=shape), jnp.bfloat16)

    weights = (
        integers(TOKENS, HIDDEN),  # hidden
        normal(HIDDEN, 128, dtype=jnp.bfloat16),  # gate
        normal(128, 384, dtype=jnp.bfloat16),  # feature gate
        integers(HIDDEN, 1664),  # projection
        normal(4, 3, 3, 128),  # conv filter
        normal(3, 1),  # A log
        normal(3, 128),  # dt bias
        normal(8, 128, dtype=jnp.bfloat16),  # norm
    )
    rec = normal(*lead, 3, 128, 128)
    hist = normal(*lead, 3, 3, 3, 128, dtype=jnp.bfloat16)
    return weights, rec, hist


def carried(hist, weights, start, length):
    hidden, projection = weights[0], weights[3]
    projected = jnp.dot(hidden, projection, preferred_element_type=jnp.float32)
    projected = projected.astype(jnp.bfloat16)
    qkv = projected[:, : 3 * 384].reshape(TOKENS, 3, 3, 128)
    return jnp.concatenate((hist, qkv[start : start + length]))[-3:]


def packed_kda(weights, rec, hist, descriptor):
    count = rec.shape[0]

    def body(length_ref, segment_ref, *refs):
        # 10 inputs and 3 outputs, then the 2 checkpoint outputs and the scratch.
        operands, checkpoint, scratch = refs[:13], refs[13:15], refs[15:]
        kernels.kda_prefill_tiled_body(
            length_ref,
            *operands,
            *scratch,
            tile=TILE,
            decay_lower_bound=-5.0,
            epsilon=1e-5,
            checkpoint=(None, *checkpoint),
            segments=(segment_ref, count),
        )

    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    states = (
        jax.ShapeDtypeStruct(rec.shape, jnp.float32),
        jax.ShapeDtypeStruct(hist.shape, jnp.bfloat16),
    )
    return pl.pallas_call(
        body,
        out_shape=(jax.ShapeDtypeStruct((TOKENS, 384), jnp.bfloat16), *states, *states),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[hbm] * 10,
            out_specs=[hbm] * 5,
            scratch_shapes=kernels.kda_prefill_tiled_scratch_shapes(
                HIDDEN, TILE, *weights[1:]
            ),
        ),
        interpret=True,
    )(jnp.array([TOKENS], jnp.int32), descriptor, *weights, rec, hist)


@pytest.mark.parametrize(
    "length",
    [
        pytest.param(384, id="full-tiles"),
        pytest.param(300, id="partial-tile"),
        pytest.param(257, id="partial-tile-under-3-rows"),
        pytest.param(130, id="partial-then-padding-tile"),
    ],
)
def test_tiled_kda_carries_last_three_real_rows(length):
    weights, rec, hist = kda_inputs(length)
    run = kernels.make_kda_tiled(seq_len=TOKENS, tile=TILE, interpret=True)
    _, _, hist_out = run(*weights, rec, hist, jnp.array([length], jnp.int32))
    np.testing.assert_array_equal(hist_out, carried(hist, weights, 0, length))


def test_packed_kda_carries_and_checkpoints_each_sequence():
    weights, rec, hist = kda_inputs(0, lead=(2,))
    # 200 rows ending mid-tile, then 2 rows after the incoming history.
    offsets, lengths, positions = (0, 256), (200, 2), (128, 0)
    descriptor = jnp.array(offsets + lengths + positions, jnp.int32)
    _, _, hist_out, _, hist_at = packed_kda(weights, rec, hist, descriptor)
    for s in range(2):
        final = carried(hist[s], weights, offsets[s], lengths[s])
        at_checkpoint = carried(hist[s], weights, offsets[s], positions[s])
        np.testing.assert_array_equal(hist_out[s], final)
        np.testing.assert_array_equal(hist_at[s], at_checkpoint)
