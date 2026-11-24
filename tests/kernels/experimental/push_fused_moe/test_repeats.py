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
"""repeats_previous_row against the XLA compare it replaces, bit for bit.

On TPU the kernel runs compiled; elsewhere in Pallas's TPU interpret mode.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.push_fused_moe.repeats import (
    repeats_previous_row,
)


def _rows(t, seed, blocks=40):
    """FP8 bytes [t, blocks, 128] with runs, including runs across 256- and
    96-row block edges, a row differing only in one byte, signed zeros, NaN."""
    rng = np.random.default_rng(seed)
    q = rng.integers(0, 256, size=(t, blocks, 128), dtype=np.uint8)
    for lo, hi in [(0, 4), (5, 9), (t // 3, t // 2), (t - t // 4, t)]:
        q[lo:hi] = q[lo]
    for edge in (96, 256):
        if t > edge + 2:
            q[edge - 2 : edge + 2] = q[edge - 2]
    if t > 300:
        q[300:303] = q[299]
        q[301, blocks - 1, 127] ^= 1
        q[310, 0, 0], q[311] = 0x80, q[310]
        q[311, 0, 0] = 0x00
        q[320, 3, 3], q[321] = 0x7F, q[320]
    return q


def _interpret():
    return False if jax.default_backend() == "tpu" else pltpu.InterpretParams()


# hidden 5120 (the served 40 lane blocks) at every bucket, and 7168 / 5632
# (56 / 44 blocks: words per row not a multiple of the sublane tile).
@pytest.mark.parametrize(
    "t,blocks", [(2048, 40), (288, 40), (240, 40), (48, 40), (512, 56), (256, 44)]
)
def test_repeats_previous_row_matches_xla(t, blocks):
    q = jnp.asarray(_rows(t, t, blocks)).view(jnp.float8_e4m3fn)
    got = jax.jit(lambda v: repeats_previous_row(v, interpret=_interpret()))(q)
    bits = lax.bitcast_convert_type(q, jnp.int8).reshape(t, -1)
    want = np.concatenate([[False], np.asarray(jnp.all(bits[1:] == bits[:-1], 1))])
    np.testing.assert_array_equal(np.asarray(got), want)
    assert want.sum() > 0 and (~want).sum() > 1
