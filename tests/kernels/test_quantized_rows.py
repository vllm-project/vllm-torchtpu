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
"""CPU tests of the quantized row format gmm_v2 reads (quantized_rows)."""

import functools

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.megablox import quantized_rows
from vllm_torchtpu.kernels.megablox.quantized_rows import (
    ROWS_PER_WORD,
    SCALE_COLS,
    decode_scales,
    encode_scales,
    quantize_block,
)


def test_scales_round_trip_bit_exact():
    rng = np.random.default_rng(0)
    rows = 64
    # Every bf16 bit pattern class: zero, subnormal, normal, max, inf, NaN.
    special = np.array([0x0000, 0x0001, 0x007F, 0x0080, 0x3F80, 0x7F7F, 0x7F80, 0x7FC0])
    bits = [rng.integers(0, 1 << 16, rows, dtype=np.uint16) for _ in range(7)]
    bits.append(np.resize(special, rows).astype(np.uint16))
    scales = [jnp.asarray(b.view(jnp.bfloat16)[:, None]) for b in bits]

    cols = encode_scales(scales, jnp.float8_e4m3fn)
    assert cols.shape == (rows, SCALE_COLS)
    assert not np.asarray(cols).view(np.uint8)[:, 2 * len(scales) :].any()
    for b, decoded in zip(bits, decode_scales(cols, len(scales))):
        np.testing.assert_array_equal(np.asarray(decoded).view(np.uint16)[:, 0], b)


@pytest.mark.parametrize("rows", [32, 96, 256, 512])
def test_quantize_rows_is_quantize_block(monkeypatch, rows):
    """quantize_rows' words hold exactly what quantize_block computes."""
    # The TPU interpreter cannot bitcast refs; the generic one can.
    monkeypatch.setattr(
        quantized_rows.pl,
        "pallas_call",
        functools.partial(quantized_rows.pl.pallas_call, interpret=True),
    )
    size_k, block = 1024, 512
    rng = np.random.default_rng(rows)
    x = rng.standard_normal((rows, size_k)) * np.exp(rng.uniform(-8, 6, (rows, 1)))
    x[3] = 0  # a zero row: zero scales
    x[5, :block] = 0  # a zero block
    x = jnp.asarray(x, jnp.bfloat16)

    words = np.asarray(
        quantized_rows.quantize_rows.__wrapped__(
            x, block_size=block, q_dtype=jnp.float8_e4m3fn
        )
    ).view(np.uint32)
    assert words.shape == (rows // ROWS_PER_WORD, size_k + SCALE_COLS)
    # Row 4 * w + j in byte j of word row w.
    shifts = 8 * np.arange(ROWS_PER_WORD, dtype=np.uint32)
    row_bytes = ((words[:, None] >> shifts[:, None]) & 0xFF).astype(np.uint8)
    row_bytes = row_bytes.reshape(rows, -1)

    cols = jnp.asarray(row_bytes[:, size_k:]).view(jnp.float8_e4m3fn)
    scales = decode_scales(cols, size_k // block)
    for b in range(size_k // block):
        want_q, want_scale = quantize_block(
            x[:, b * block : (b + 1) * block], jnp.float8_e4m3fn
        )
        np.testing.assert_array_equal(
            row_bytes[:, b * block : (b + 1) * block], np.asarray(want_q).view(np.uint8)
        )
        np.testing.assert_array_equal(
            np.asarray(scales[b]).view(np.uint16),
            np.asarray(want_scale).view(np.uint16),
        )
