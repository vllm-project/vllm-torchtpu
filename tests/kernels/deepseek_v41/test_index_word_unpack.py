# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed keys retain record order and BF16 conversion semantics."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v41.index_word_scores import unpack_word_rows


@pytest.mark.cpu_test
@pytest.mark.parametrize("rows", [8, 512])
def test_unpack_preserves_record_order_and_bf16_bits(rows):
    # At 512 rows, each half covers every BF16 pattern, including NaNs,
    # infinities, subnormals and signed zero. Distinct halves expose swaps.
    low = np.arange(rows * 128, dtype=np.uint32).reshape(2 * rows, 64)
    high = np.uint32(65535) - low
    words = (low | (high << 16)).reshape(rows, 128).view(np.int32)
    expected_bits = np.concatenate((low, high), axis=1) << 16
    # The contract includes FP32-to-BF16 NaN conversion, not raw BF16 bitcast.
    expected = jax.lax.bitcast_convert_type(
        jnp.asarray(expected_bits), jnp.float32
    ).astype(jnp.bfloat16)

    def kernel(words_ref, out_ref, scratch):
        out_ref[...] = unpack_word_rows(words_ref[...], scratch)

    actual = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((2 * rows, 128), jnp.bfloat16),
        scratch_shapes=(pltpu.VMEM((2 * rows, 128), jnp.int32),),
        interpret=True,
    )(jnp.asarray(words))
    np.testing.assert_array_equal(
        np.asarray(actual.view(jnp.uint16)),
        np.asarray(expected.view(jnp.uint16)),
    )
