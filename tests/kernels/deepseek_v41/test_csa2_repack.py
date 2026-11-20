# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Byte-preserving main KV repacking and the tile reads shared by QK and PV."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v41.csa2.csa2 import (
    _load_main_tile,
    _repack_main_words,
)


@pytest.mark.cpu_test
@pytest.mark.parametrize("pattern", ["random", "byte_positions"])
def test_main_repack_preserves_bytes_across_tile_boundaries(pattern):
    shape = (2, 512, 128, 4)
    if pattern == "random":
        data = np.random.default_rng(31).integers(0, 256, shape, dtype=np.uint8)
    else:
        i, row, lane, group = np.indices(shape, dtype=np.uint32)
        data = ((i * 137 + row * 19 + lane * 7 + group * 53) & 255).astype(np.uint8)
    words = sum(data[..., g].astype(np.uint32) << (8 * g) for g in range(4))
    expected = data.transpose(0, 1, 3, 2).reshape(2, 512, 512)

    def kernel(src, out, scratch):
        for i in range(2):
            for row in range(0, 512, 8):
                rows = pl.ds(row, 8)
                scratch[i, rows, :] = _repack_main_words(src[i, rows, :])
            for row in range(0, 512, 256):
                for column in range(0, 512, 256):
                    rows, columns = pl.ds(row, 256), pl.ds(column, 256)
                    tile = _load_main_tile(scratch, i, rows, columns)
                    out[i, rows, columns] = lax.bitcast_convert_type(tile, jnp.uint8)

    result = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(expected.shape, jnp.uint8),
        scratch_shapes=(pltpu.VMEM(words.shape, jnp.int32),),
        interpret=jax.default_backend() != "tpu",
    )(jnp.asarray(words.view(np.int32)))
    np.testing.assert_array_equal(result, expected)
